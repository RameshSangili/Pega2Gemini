from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Logging – verbose, timestamped, goes straight to Render's log stream
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="0.1.0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
TOOLS: list = [
    {
        "name": "eligibility_check",
        "description": "Check loan eligibility based on income, credit score, employment and amount.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "income": {"type": "number", "description": "Monthly income in USD"},
                "credit_score": {"type": "integer", "description": "Credit score 300-850"},
                "employment": {"type": "string", "description": "employed | self-employed | unemployed"},
                "loan_amount": {"type": "number", "description": "Requested loan amount in USD"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    },
    {
        "name": "loan_recommendation",
        "description": "Recommend the best loan product for the applicant profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "Purpose of the loan"},
                "amount": {"type": "number", "description": "Loan amount in USD"},
                "credit_score": {"type": "integer", "description": "Applicant credit score"},
                "tenure_months": {"type": "integer", "description": "Repayment period in months"},
            },
            "required": ["purpose", "amount", "credit_score"],
        },
    },
    {
        "name": "credit_summary",
        "description": "Summarise credit profile and highlight key risk factors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credit_score": {"type": "integer", "description": "Credit score"},
                "outstanding_debt": {"type": "number", "description": "Total outstanding debt in USD"},
                "payment_history": {"type": "string", "description": "good | average | poor"},
                "num_open_accounts": {"type": "integer", "description": "Number of open credit accounts"},
            },
            "required": ["credit_score"],
        },
    },
]

# ---------------------------------------------------------------------------
# Per-session SSE queues
# ---------------------------------------------------------------------------
_sessions: dict = {}


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY is not set!")
        return "ERROR: GEMINI_API_KEY environment variable is not set."
    log.debug("Calling Gemini, prompt length=%d chars", len(prompt))
    t0 = time.time()
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
            resp.raise_for_status()
            data = resp.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"]
            log.info("Gemini responded in %.2fs, reply length=%d chars", time.time() - t0, len(result))
            return result
    except Exception as exc:
        log.error("Gemini call failed after %.2fs: %s", time.time() - t0, exc)
        return f"Gemini call failed: {exc}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Dispatching tool=%s args=%s", tool_name, arguments)
    args_json = json.dumps(arguments, indent=2)
    prompts = {
        "eligibility_check": (
            "You are a loan underwriting assistant. "
            "Given the following applicant details decide whether they are eligible for a loan.\n\n"
            + args_json + "\n\nProvide a clear YES or NO decision followed by a brief explanation."
        ),
        "loan_recommendation": (
            "You are a loan advisor. Recommend the best loan product for this applicant.\n\n"
            + args_json
        ),
        "credit_summary": (
            "You are a credit analyst. Summarise the credit profile and highlight top 3 risk factors.\n\n"
            + args_json
        ),
    }
    prompt = prompts.get(tool_name)
    if not prompt:
        log.warning("Unknown tool requested: %s", tool_name)
        return f"Unknown tool: {tool_name}"
    return await call_gemini(prompt)


# ---------------------------------------------------------------------------
# MCP JSON-RPC builder
# ---------------------------------------------------------------------------
async def build_response(message: dict):
    method: str = message.get("method", "")
    msg_id = message.get("id")
    log.info(">>> MCP request method=%s id=%s", method, msg_id)

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "20250326",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1"},
            },
        }
        log.info("<<< Sending initialize response: %s", json.dumps(resp))
        return resp

    if method == "notifications/initialized":
        log.info("<<< notifications/initialized received (no response)")
        return None

    if method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
        log.info("<<< Sending tools/list with %d tools", len(TOOLS))
        return resp

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        log.info("<<< tools/call tool=%s", tool_name)
        result_text = await dispatch_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": result_text}], "isError": False},
        }

    if method == "ping":
        log.info("<<< pong")
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    log.warning("Unknown method=%s id=%s", method, msg_id)
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None


# ---------------------------------------------------------------------------
# DIAGNOSTIC MIDDLEWARE – logs every request/response with full headers+body
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    # Log request
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8", errors="replace")[:2000]

    log.info(
        "==> REQ [%s] %s %s | headers=%s | body=%s",
        req_id,
        request.method,
        request.url,
        dict(request.headers),
        body_str if body_str else "<empty>",
    )

    # Reconstruct request body so downstream can re-read it
    async def receive():
        return {"type": "http.request", "body": body_bytes}

    request._receive = receive

    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000

    log.info(
        "<== RES [%s] status=%d elapsed=%.1fms",
        req_id,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# GET /mcp/ – SSE stream
# ---------------------------------------------------------------------------
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue
    client_host = request.client.host if request.client else "unknown"

    log.info(
        "SSE CONNECT session=%s client=%s | total_sessions=%d",
        session_id, client_host, len(_sessions),
    )

    async def event_generator():
        try:
            t_connect = time.time()

            # 1. Send endpoint event
            endpoint_event = f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"
            log.info("SSE [%s] sending endpoint event -> /mcp/messages/?session_id=%s", session_id, session_id)
            yield endpoint_event

            # 2. Immediately send initialize result
            init_resp = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "20250326",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "Pega2Gemini", "version": "1"},
                },
            }
            log.info("SSE [%s] sending proactive initialize result", session_id)
            yield f"data: {json.dumps(init_resp)}\n\n"

            # 3. Immediately send tools/list result
            tools_resp = {"jsonrpc": "2.0", "id": 2, "result": {"tools": TOOLS}}
            log.info("SSE [%s] sending proactive tools/list (%d tools)", session_id, len(TOOLS))
            yield f"data: {json.dumps(tools_resp)}\n\n"

            log.info("SSE [%s] handshake complete in %.1fms – waiting for tool calls",
                     session_id, (time.time() - t_connect) * 1000)

            # 4. Keep alive, drain tool-call responses
            last_ping = time.time()
            while not await request.is_disconnected():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=10.0)
                    log.info("SSE [%s] pushing response to stream: %s", session_id, json.dumps(msg)[:300])
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    now = time.time()
                    log.debug("SSE [%s] keep-alive (connected %.0fs)", session_id, now - t_connect)
                    last_ping = now
                    yield ": keep-alive\n\n"

        except Exception as exc:
            log.error("SSE [%s] stream error: %s", session_id, exc, exc_info=True)
        finally:
            _sessions.pop(session_id, None)
            log.info("SSE [%s] DISCONNECTED | remaining_sessions=%d", session_id, len(_sessions))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /mcp/messages/ – synchronous JSON-RPC handler
# ---------------------------------------------------------------------------
@app.post("/mcp/messages/")
async def mcp_messages(request: Request, session_id: str = ""):
    t0 = time.time()
    client_host = request.client.host if request.client else "unknown"
    log.info("POST /mcp/messages/ session=%s client=%s", session_id, client_host)

    try:
        body = await request.json()
    except Exception as exc:
        log.error("POST parse error: %s", exc)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    log.info("POST body: %s", json.dumps(body)[:1000])

    messages = body if isinstance(body, list) else [body]
    responses = []

    for msg in messages:
        resp = await build_response(msg)
        if resp is not None:
            responses.append(resp)
            # Also push onto the SSE queue for completeness
            q = _sessions.get(session_id) or (next(iter(_sessions.values())) if _sessions else None)
            if q:
                await q.put(resp)
                log.info("POST queued response to SSE session=%s", session_id)
            else:
                log.warning("POST no active SSE session found! session_id=%s known=%s",
                            session_id, list(_sessions.keys()))

    elapsed = (time.time() - t0) * 1000
    log.info("POST handled in %.1fms, returning %d response(s)", elapsed, len(responses))

    if len(responses) == 1:
        return JSONResponse(responses[0], status_code=200)
    if len(responses) > 1:
        return JSONResponse(responses, status_code=200)

    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# Debug endpoint – dump current state
# ---------------------------------------------------------------------------
@app.get("/debug")
async def debug():
    return {
        "active_sessions": list(_sessions.keys()),
        "session_count": len(_sessions),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools_count": len(TOOLS),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    log.info("Health check called")
    return {"status": "ok", "service": "Pega2Gemini MCP", "sessions": len(_sessions)}
