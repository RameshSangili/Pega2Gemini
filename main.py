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
# Streamable HTTP MCP Server (protocol version 2025-03-26)
#
# KEY INSIGHT: Pega uses the NEW Streamable HTTP transport, not old SSE.
# In Streamable HTTP:
# - ALL messages go to POST /mcp (single endpoint)
# - GET /mcp opens an optional SSE stream for server-push only
# - The client POSTs initialize directly to /mcp, not /mcp/messages
# - Response can be JSON or SSE depending on Accept header
# ---------------------------------------------------------------------------

app = FastAPI(title="Pega2Gemini", version="0.5.0", redirect_slashes=False)

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
# ---------------------------------------------------------------------------
async def handle_message(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id")
    log.info("MCP method=%r id=%r", method, msg_id)

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
        return None # notification, no response

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
# POST /mcp — Streamable HTTP: ALL client->server messages come here
#
# The client sends Accept: application/json, text/event-stream
# We respond with JSON for simple request/response.
# For streaming tool calls we could return SSE, but JSON is fine for Pega.
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

    log.info("POST returning %d response(s)", len(responses))

    if len(responses) == 1:
        # Check if client wants SSE response format
        if "text/event-stream" in accept and "application/json" not in accept:
            # Wrap in SSE format
            data = json.dumps(responses[0])
            async def sse_gen():
                yield f"data: {data}\n\n"
            return StreamingResponse(sse_gen(), media_type="text/event-stream",
                                     headers={"X-Accel-Buffering": "no"})
        return JSONResponse(responses[0], status_code=200)

    if len(responses) > 1:
        return JSONResponse(responses, status_code=200)

    # Notification only
    return JSONResponse({}, status_code=202)


# ---------------------------------------------------------------------------
# GET /mcp — Optional SSE stream for server-push (Streamable HTTP spec)
# Pega may open this to receive async notifications from server.
# We keep it alive with pings. No tools handshake happens here anymore.
# ---------------------------------------------------------------------------
async def _handle_get(request: Request):
    session_id = str(uuid.uuid4())
    log.info("GET /mcp (SSE stream) session=%s", session_id)

    async def event_generator():
        try:
            log.info("SSE [%s] opened", session_id)
            t = time.time()
            while True:
                if await request.is_disconnected():
                    log.info("SSE [%s] disconnected after %.0fs", session_id, time.time() - t)
                    break
                await asyncio.sleep(10)
                yield ": ping\n\n"
        except Exception as exc:
            log.error("SSE [%s] error: %s", session_id, exc)
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
# Single /mcp endpoint — handles both GET and POST (Streamable HTTP spec)
# Also registered without trailing slash to avoid 307 redirects
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

# Keep old /mcp/messages routes for backward compat in case Pega uses them
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
        "version": "0.5.0",
        "transport": "Streamable HTTP (2025-03-26)",
        "key_change": "POST /mcp handles ALL JSON-RPC — no separate /mcp/messages",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools": [t["name"] for t in TOOLS],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "version": "0.5.0"}

