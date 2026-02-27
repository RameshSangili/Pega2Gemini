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
# CRITICAL FIX: Disable FastAPI's automatic redirect from /mcp to /mcp/
# Cloud Run was returning 307 for /mcp -> /mcp/ and Apache-CXF (Pega's
# Java HTTP client) does NOT follow 307 redirects on SSE connections.
# ---------------------------------------------------------------------------
app = FastAPI(title="Pega2Gemini", version="0.4.0", redirect_slashes=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

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

        log.info("==> REQ [%s] %s %s?%s ua=%s",
                 req_id, method, path, qs, headers.get("user-agent", "?"))

        async def logging_send(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info("<== RES [%s] status=%s elapsed=%.1fms",
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

# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------
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
    log.info("MCP IN method=%r id=%r", method, msg_id)

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
        log.warning("Unknown method: %r", method)
        resp = {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    log.info("MCP OUT %s", json.dumps(resp)[:300])
    return resp


# ---------------------------------------------------------------------------
# SSE stream handler (shared logic)
# ---------------------------------------------------------------------------
async def _sse_handler(request: Request, session_id: str):
    log.info("SSE OPEN session=%s path=%s", session_id, request.url.path)

    async def event_generator():
        try:
            # Send the endpoint event pointing to the messages path
            # Use the same path style Pega called us with (no trailing slash)
            endpoint_url = f"/mcp/messages?session_id={session_id}"
            event = f"event: endpoint\ndata: {endpoint_url}\n\n"
            log.info("SSE [%s] sending endpoint: %s", session_id, endpoint_url)
            yield event

            t_start = time.time()
            ping_count = 0
            while True:
                if await request.is_disconnected():
                    log.info("SSE [%s] disconnected after %.1fs", session_id, time.time() - t_start)
                    break
                await asyncio.sleep(10)
                ping_count += 1
                log.debug("SSE [%s] ping #%d", session_id, ping_count)
                yield ": ping\n\n"

        except asyncio.CancelledError:
            log.info("SSE [%s] cancelled", session_id)
        except Exception as exc:
            log.error("SSE [%s] error: %s", session_id, exc, exc_info=True)
        finally:
            log.info("SSE [%s] closed", session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# POST handler (shared logic) — fully stateless
# ---------------------------------------------------------------------------
async def _post_handler(request: Request, session_id: str):
    t0 = time.perf_counter()
    log.info("POST session=%s path=%s", session_id, request.url.path)

    try:
        raw = await request.body()
        log.info("POST body: %s", raw.decode("utf-8", errors="replace")[:800])
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
        return JSONResponse(responses[0], status_code=200)
    if len(responses) > 1:
        return JSONResponse(responses, status_code=200)
    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# Routes — registered for BOTH with and without trailing slash
# This prevents the 307 redirect that kills Apache-CXF SSE connections
# ---------------------------------------------------------------------------

@app.get("/mcp")
async def mcp_sse_noslash(request: Request):
    """SSE endpoint without trailing slash — what Pega actually calls"""
    session_id = str(uuid.uuid4())
    return await _sse_handler(request, session_id)

@app.get("/mcp/")
async def mcp_sse_slash(request: Request):
    """SSE endpoint with trailing slash — fallback"""
    session_id = str(uuid.uuid4())
    return await _sse_handler(request, session_id)

@app.post("/mcp/messages")
async def mcp_post_noslash(request: Request, session_id: str = ""):
    """POST without trailing slash"""
    return await _post_handler(request, session_id)

@app.post("/mcp/messages/")
async def mcp_post_slash(request: Request, session_id: str = ""):
    """POST with trailing slash"""
    return await _post_handler(request, session_id)


# ---------------------------------------------------------------------------
# Debug & Health
# ---------------------------------------------------------------------------
@app.get("/debug")
async def debug():
    return {
        "version": "0.4.0",
        "fix": "redirect_slashes=False + dual routes for /mcp and /mcp/",
        "root_cause": "Apache-CXF/4.1.1 does not follow 307 redirect on SSE",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools": [t["name"] for t in TOOLS],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "version": "0.4.0"}