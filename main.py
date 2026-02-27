from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from starlette.types import ASGIApp, Scope, Receive, Send

# ---------------------------------------------------------------------------
# Logging — every byte in/out is logged
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="0.3.0")

# ---------------------------------------------------------------------------
# Middleware — log ALL request headers and body
# ---------------------------------------------------------------------------
class LoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        qs = scope.get("query_string", b"").decode()
        headers = dict(scope.get("headers", []))
        headers_str = {k.decode(): v.decode() for k, v in headers.items()}

        log.info("=" * 60)
        log.info("REQ [%s] %s %s?%s", req_id, method, path, qs)
        log.info("REQ [%s] headers: %s", req_id, json.dumps(headers_str))

        # Buffer body so we can log it
        body_chunks = []
        original_receive = receive

        async def logging_receive():
            msg = await original_receive()
            if msg.get("type") == "http.request":
                chunk = msg.get("body", b"")
                body_chunks.append(chunk)
                log.info("REQ [%s] body: %s", req_id, chunk.decode("utf-8", errors="replace"))
            return msg

        async def logging_send(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                res_headers = dict(message.get("headers", []))
                res_headers_str = {k.decode(): v.decode() for k, v in res_headers.items()}
                log.info("RES [%s] status=%s elapsed=%.1fms headers=%s",
                         req_id, message["status"], elapsed, json.dumps(res_headers_str))
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    log.info("RES [%s] body: %s", req_id, body.decode("utf-8", errors="replace")[:500])
            await send(message)

        await self.app(scope, logging_receive, logging_send)

app.add_middleware(LoggingMiddleware)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

TOOLS = [
    {
        "name": "eligibility_check",
        "description": "Check loan eligibility based on applicant profile.",
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
        "description": "Recommend the best loan product.",
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
        "description": "Summarise credit profile and risk factors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credit_score": {"type": "integer", "description": "Credit score"},
                "outstanding_debt": {"type": "number", "description": "Outstanding debt in USD"},
                "payment_history": {"type": "string", "description": "good | average | poor"},
                "num_open_accounts": {"type": "integer", "description": "Number of open accounts"},
            },
            "required": ["credit_score"],
        },
    },
]

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY not set."
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        log.error("Gemini failed: %s", exc)
        return f"Gemini error: {exc}"

async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Tool: %s args: %s", tool_name, arguments)
    prompt = (
        f"You are an expert loan officer. Tool: {tool_name}. "
        f"Input: {json.dumps(arguments)}. Give a concise professional answer."
    )
    return await call_gemini(prompt)

# ---------------------------------------------------------------------------
# MCP Response Builder
# ---------------------------------------------------------------------------
async def build_response(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id")
    log.info("MCP IN method=%r id=%r full=%s", method, msg_id, json.dumps(message))

    resp = None

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1.0.0"},
            },
        }

    elif method == "notifications/initialized":
        log.info("MCP notifications/initialized — no response")
        return None

    elif method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    elif method == "tools/call":
        params = message.get("params", {})
        res_text = await dispatch_tool(params.get("name", ""), params.get("arguments", {}))
        resp = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": res_text}], "isError": False},
        }

    elif method == "ping":
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    else:
        log.warning("MCP unknown method=%r", method)
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    log.info("MCP OUT %s", json.dumps(resp) if resp else "None")
    return resp


# ---------------------------------------------------------------------------
# GET /mcp/ SSE stream
#
# IMPORTANT: We log every single byte yielded so we can see exactly
# what Pega's Java client receives and compare with the MCP spec.
# ---------------------------------------------------------------------------
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    client = request.client.host if request.client else "unknown"
    accept = request.headers.get("accept", "")
    ua = request.headers.get("user-agent", "")

    log.info("SSE OPEN session=%s client=%s accept=%r ua=%r", session_id, client, accept, ua)
    log.info("SSE ALL headers: %s", dict(request.headers))

    async def event_generator():
        try:
            # ── Event 1: endpoint ────────────────────────────────────────
            # MCP SSE spec: first event MUST be 'endpoint' telling client
            # where to POST messages. Some clients need trailing slash, some don't.
            msg_url = f"/mcp/messages/?session_id={session_id}"
            line1 = f"event: endpoint\ndata: {msg_url}\n\n"
            log.info("SSE [%s] YIELD endpoint event: %r", session_id, line1)
            yield line1

            # Keep alive loop
            t_start = time.time()
            ping_count = 0
            while True:
                if await request.is_disconnected():
                    log.info("SSE [%s] client disconnected after %.1fs pings=%d",
                             session_id, time.time() - t_start, ping_count)
                    break

                await asyncio.sleep(10)
                ping_count += 1
                # SSE comment line (keep-alive) — does NOT trigger onmessage
                ping_line = ": ping\n\n"
                log.debug("SSE [%s] ping #%d", session_id, ping_count)
                yield ping_line

        except asyncio.CancelledError:
            log.info("SSE [%s] cancelled", session_id)
        except Exception as exc:
            log.error("SSE [%s] error: %s", session_id, exc, exc_info=True)
        finally:
            log.info("SSE [%s] stream ended", session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Some MCP clients need this
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# POST /mcp/messages/ Fully stateless JSON-RPC handler
# ---------------------------------------------------------------------------
@app.post("/mcp/messages/")
async def mcp_message(request: Request, session_id: str = ""):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    ct = request.headers.get("content-type", "")
    log.info("POST session=%s client=%s content-type=%r", session_id, client, ct)

    try:
        raw = await request.body()
        log.info("POST raw body bytes=%d: %s", len(raw), raw.decode("utf-8", errors="replace"))
        body = json.loads(raw)
    except Exception as exc:
        log.error("POST parse error: %s", exc)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": f"Parse error: {exc}"}},
            status_code=400,
        )

    messages = body if isinstance(body, list) else [body]
    responses = []

    for msg in messages:
        resp = await build_response(msg)
        if resp is not None:
            responses.append(resp)

    elapsed = (time.perf_counter() - t0) * 1000
    log.info("POST done %.1fms responses=%d", elapsed, len(responses))

    if len(responses) == 1:
        log.info("POST returning 200: %s", json.dumps(responses[0]))
        return JSONResponse(responses[0], status_code=200)
    if len(responses) > 1:
        log.info("POST returning 200 batch: %d items", len(responses))
        return JSONResponse(responses, status_code=200)

    log.info("POST returning 202 (notification only)")
    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# GET /mcp/messages/ — Some MCP clients do a GET first to negotiate
# ---------------------------------------------------------------------------
@app.get("/mcp/messages/")
async def mcp_messages_get(request: Request, session_id: str = ""):
    log.info("GET /mcp/messages/ session=%s headers=%s", session_id, dict(request.headers))
    return JSONResponse({"status": "ok", "session_id": session_id})


# ---------------------------------------------------------------------------
# Raw SSE test — visit this in browser to verify SSE format
# /mcp-test/ sends a fake initialize response immediately so you can
# verify Pega can parse it
# ---------------------------------------------------------------------------
@app.get("/mcp-test/")
async def mcp_test(request: Request):
    """
    Test endpoint: open this in browser or curl to verify SSE format.
    Immediately sends endpoint event + fake initialize response.
    Use this to check if Pega's parser can handle our SSE format.
    """
    session_id = "test-" + str(uuid.uuid4())[:8]

    async def gen():
        yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"
        await asyncio.sleep(0.1)

        init_resp = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1.0.0"},
            },
        }
        yield f"event: message\ndata: {json.dumps(init_resp)}\n\n"
        await asyncio.sleep(0.1)

        tools_resp = {"jsonrpc": "2.0", "id": 2, "result": {"tools": TOOLS}}
        yield f"event: message\ndata: {json.dumps(tools_resp)}\n\n"

        for i in range(30):
            await asyncio.sleep(5)
            yield f": ping {i}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------
@app.get("/debug")
async def debug():
    return {
        "version": "0.3.0",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools": [t["name"] for t in TOOLS],
        "endpoints": {
            "sse": "GET /mcp/",
            "messages": "POST /mcp/messages/?session_id=<id>",
            "test_sse": "GET /mcp-test/",
            "health": "GET /health",
            "debug": "GET /debug",
        },
        "note": "Check /mcp-test/ in browser to validate SSE format",
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "version": "0.3.0"}