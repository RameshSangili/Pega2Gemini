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
# Pega2Gemini MCP Server for Google Cloud Run
# ---------------------------------------------------------------------------

app = FastAPI(title="Pega2Gemini", version="0.7.0", redirect_slashes=False)

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
                "income": {"type": "number"},
                "credit_score": {"type": "integer"},
                "loan_amount": {"type": "number"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    }
]

async def handle_message(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id") 
    
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
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    return None

# ---------------------------------------------------------------------------
# SSE Endpoint (GET /mcp)
# ---------------------------------------------------------------------------
@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info("SSE OPEN session=%s", session_id)

    async def event_generator():
        try:
            # Step 1: Push 2KB padding to clear Cloud Run/GFE buffer
            yield GFE_FLUSH_PADDING
            
            # Step 2: Send the actual endpoint Pega needs
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info("SSE [%s] endpoint sent with padding", session_id)
            
            while True:
                if await request.is_disconnected():
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
            "Cache-Control": "no-cache, no-transform", # NO-TRANSFORM is the fix
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ---------------------------------------------------------------------------
# Message Endpoint (POST /mcp/messages)
# ---------------------------------------------------------------------------
@app.post("/mcp/messages")
async def mcp_message(request: Request):
    session_id = request.query_params.get("session_id", "unknown")
    try:
        body = await request.json()
        log.info("POST session=%s method=%s", session_id, body.get("method"))
        
        # Handle batch or single message
        msgs = body if isinstance(body, list) else [body]
        responses = []
        for m in msgs:
            r = await handle_message(m)
            if r: responses.append(r)
            
        if len(responses) == 1:
            return JSONResponse(responses[0])
        return JSONResponse(responses) if responses else JSONResponse({"status":"ok"}, status_code=202)
        
    except Exception as exc:
        log.error("POST error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
