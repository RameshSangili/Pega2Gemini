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
# Pega2Gemini MCP Server for Google Cloud Run (Protocol 2025-03-26)
# ---------------------------------------------------------------------------

app = FastAPI(title="Pega2Gemini", version="0.9.0", redirect_slashes=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding to force Google Cloud Run (GFE) to flush the buffer immediately
GFE_FLUSH_PADDING = ": " + (" " * 2048) + "\n\n"

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
        async def logging_send(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info("<== [%s] %s %s status=%s elapsed=%.1fms",
                         req_id, method, path, message["status"], elapsed)
            await send(message)
        await self.app(scope, receive, logging_send)

app.add_middleware(LoggingMiddleware)

# ---------------------------------------------------------------------------
# Config & Gemini logic
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = "https://generativelanguage.googleapis.com"

TOOLS = [
    {
        "name": "eligibility_check",
        "description": "Check loan eligibility based on applicant profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "income": {"type": "number", "description": "Monthly income in USD"},
                "credit_score": {"type": "integer", "description": "Credit score 300-850"},
                "loan_amount": {"type": "number", "description": "Requested loan amount in USD"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    }
]

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
# MCP JSON-RPC Handler
# ---------------------------------------------------------------------------
async def handle_message(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id")
    log.info("Processing MCP method=%r id=%r", method, msg_id)

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

    # CRITICAL: Handle the "I'm ready" notification from Pega
    if method == "notifications/initialized":
        log.info("MCP Handshake finalized by Pega client.")
        return None 

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "unknown")
        args = params.get("arguments", {})
        prompt = f"You are a loan officer. Tool: {tool_name}. Input: {json.dumps(args)}. Give a professional answer."
        res_text = await call_gemini(prompt)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": res_text}], "isError": False},
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# ---------------------------------------------------------------------------
# SSE & Message Routes
# ---------------------------------------------------------------------------
@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info("SSE OPEN session=%s", session_id)

    async def event_generator():
        try:
            # Step 1: Force Cloud Run GFE to flush buffer immediately
            yield GFE_FLUSH_PADDING 
            
            # Step 2: Send endpoint with absolute relative path
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info("SSE [%s] endpoint sent with padding", session_id)
            
            while True:
                if await request.is_disconnected():
                    log.info("SSE [%s] client disconnected", session_id)
                    break
                await asyncio.sleep(15)
                yield ": heartbeat\n\n"
        except Exception as exc:
            log.error("SSE error: %s", exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform", # CRITICAL: Disable GFE buffering
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/mcp/messages")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info("POST REQ: %s", body.get("method"))
        
        # Determine if it's a list (batch) or single message
        messages = body if isinstance(body, list) else [body]
        responses = []
        
        for msg in messages:
            resp = await handle_message(msg)
            if resp is not None:
                responses.append(resp)
        
        # KEEP-ALIVE: Wait slightly after 'initialize' to ensure container 
        # doesn't shut down before Pega's next notification arrives.
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5) 
            
        if len(responses) == 1:
            return JSONResponse(responses[0])
        elif len(responses) > 1:
            return JSONResponse(responses)
            
        return JSONResponse({"status": "accepted"}, status_code=202)
        
    except Exception as exc:
        log.error("POST error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP"}
