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
from starlette.types import ASGIApp, Scope, Receive, Send

# ---------------------------------------------------------------------------
# Cloud Run / GFE buffering notes:
#
# Cloud Run's Google Front End (GFE) IGNORES X-Accel-Buffering and buffers
# SSE chunks until it sees enough data. Fixes applied here:
#
# 1. Cache-Control: no-cache, no-transform ← stops GFE compression/buffering
# 2. Transfer-Encoding: chunked ← forces streaming mode
# 3. 2KB padding comment before endpoint ← flushes GFE's initial buffer
# 4. Standard MCP ping: "event: ping\ndata: {}\n\n" ← not just a comment
# 5. POST /mcp returns response wrapped in SSE if client wants event-stream
# 6. Preserve id type from request (string vs int) ← Apache-CXF strict
# ---------------------------------------------------------------------------

app = FastAPI(title="Pega2Gemini", version="0.6.0", redirect_slashes=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding to flush Cloud Run GFE's initial buffer
GFE_FLUSH_PADDING = ": " + (" " * 2046) + "\n\n"


# ---------------------------------------------------------------------------
# Logging Middleware
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
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}

        log.info("==> [%s] %s %s?%s accept=%r ua=%r",
                 req_id, method, path, qs,
                 headers.get("accept", ""),
                 headers.get("user-agent", ""))

        async def logging_send(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info("<== [%s] status=%s elapsed=%.1fms",
                         req_id, message["status"], elapsed)
            await send(message)

        await self.app(scope, receive, logging_send)

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
    log.info("Tool: %s", tool_name)
    prompt = (
        f"You are an expert loan officer. Tool: {tool_name}. "
        f"Input: {json.dumps(arguments)}. Give a concise professional answer."
    )
    return await call_gemini(prompt)

# ---------------------------------------------------------------------------
# MCP JSON-RPC handler
# FIX: Preserve id type exactly as received (Apache-CXF is strict about this)
# ---------------------------------------------------------------------------
async def handle_message(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id") # preserve type: string OR int, never cast
    log.info("MCP method=%r id=%r (type=%s)", method, msg_id, type(msg_id).__name__)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = message.get("params", {})
        res_text = await dispatch_tool(params.get("name", ""), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": res_text}], "isError": False},
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    log.warning("Unknown method: %r", method)
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None


# ---------------------------------------------------------------------------
# SSE response headers — Cloud Run GFE requires no-transform
# ---------------------------------------------------------------------------
SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform", # no-transform stops GFE buffering
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no", # for nginx-based proxies
    "Transfer-Encoding": "chunked", # force streaming
    "Access-Control-Allow-Origin": "*",
}


# ---------------------------------------------------------------------------
# GET /mcp — SSE stream
#
# FIX 1: Send 2KB padding first to flush GFE's initial buffer
# FIX 2: Use proper MCP ping format: event:ping + data:{} not just a comment
# ---------------------------------------------------------------------------
async def _handle_get(request: Request):
    session_id = str(uuid.uuid4())
    log.info("GET /mcp SSE session=%s", session_id)

    async def event_generator():
        try:
            # FIX: 2KB padding comment to force GFE to flush its buffer
            # GFE buffers until it has ~4KB before sending to client
            log.info("SSE [%s] sending GFE flush padding", session_id)
            yield GFE_FLUSH_PADDING

            # Send endpoint event
            endpoint_url = f"/mcp/messages?session_id={session_id}"
            event = f"event: endpoint\ndata: {endpoint_url}\n\n"
            log.info("SSE [%s] sending endpoint: %s", session_id, endpoint_url)
            yield event

            t_start = time.time()
            ping_count = 0
            while True:
                if await request.is_disconnected():
                    log.info("SSE [%s] disconnected after %.0fs", session_id, time.time() - t_start)
                    break
                await asyncio.sleep(15)
                ping_count += 1
                # FIX: Standard MCP ping format with data payload, not just a comment
                ping_event = "event: ping\ndata: {}\n\n"
                log.debug("SSE [%s] ping #%d", session_id, ping_count)
                yield ping_event

        except asyncio.CancelledError:
            log.info("SSE [%s] cancelled", session_id)
        except Exception as exc:
            log.error("SSE [%s] error: %s", session_id, exc, exc_info=True)
        finally:
            log.info("SSE [%s] closed", session_id)

    return StreamingResponse(event_generator(), headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# POST /mcp — ALL JSON-RPC messages (Streamable HTTP 2025-03-26)
#
# FIX: When client sends Accept: text/event-stream, wrap response in SSE.
# Pega's Apache-CXF client may send this header expecting SSE-wrapped JSON.
# ---------------------------------------------------------------------------
async def _handle_post(request: Request):
    session_id = request.query_params.get("session_id", "")
    accept = request.headers.get("accept", "")
    log.info("POST /mcp session=%s accept=%r", session_id, accept)

    try:
        raw = await request.body()
        log.info("POST body: %s", raw.decode("utf-8", errors="replace")[:1000])
        body = json.loads(raw)
    except Exception as exc:
        log.error("Parse error: %s", exc)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": f"Parse error: {exc}"}},
            status_code=400,
        )

    messages = body if isinstance(body, list) else [body]
    responses = []
    for msg in messages:
        resp = await handle_message(msg)
        if resp is not None:
            responses.append(resp)

    log.info("POST %d response(s)", len(responses))

    # No response needed (notification)
    if not responses:
        return JSONResponse({}, status_code=202)

    result = responses[0] if len(responses) == 1 else responses

    # FIX: If client wants SSE format, wrap in SSE event
    # This handles cases where Apache-CXF sends Accept: text/event-stream
    wants_sse = "text/event-stream" in accept
    wants_json = "application/json" in accept

    if wants_sse and not wants_json:
        log.info("POST returning SSE-wrapped response")
        data = json.dumps(result)
        async def sse_response():
            yield GFE_FLUSH_PADDING
            yield f"event: message\ndata: {data}\n\n"
        return StreamingResponse(sse_response(), headers=SSE_HEADERS)

    log.info("POST returning JSON response: %s", json.dumps(result)[:300])
    return JSONResponse(result, status_code=200)


# ---------------------------------------------------------------------------
# Routes — both with and without trailing slash (avoids 307 on Apache-CXF)
# ---------------------------------------------------------------------------
@app.get("/mcp")
async def mcp_get(request: Request):
    return await _handle_get(request)

@app.get("/mcp/")
async def mcp_get_slash(request: Request):
    return await _handle_get(request)

@app.post("/mcp")
async def mcp_post(request: Request):
    return await _handle_post(request)

@app.post("/mcp/")
async def mcp_post_slash(request: Request):
    return await _handle_post(request)

@app.post("/mcp/messages")
async def mcp_messages(request: Request):
    return await _handle_post(request)

@app.post("/mcp/messages/")
async def mcp_messages_slash(request: Request):
    return await _handle_post(request)


# ---------------------------------------------------------------------------
# Debug & Health
# ---------------------------------------------------------------------------
@app.get("/debug")
async def debug():
    return {
        "version": "0.6.0",
        "fixes": [
            "no-transform header stops GFE buffering",
            "2KB padding flushes GFE initial buffer",
            "SSE ping uses event:ping + data:{} not comment",
            "POST wraps in SSE when Accept: text/event-stream",
            "id type preserved exactly as received",
            "redirect_slashes=False avoids 307 on Apache-CXF",
        ],
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools": [t["name"] for t in TOOLS],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "version": "0.6.0"}

