from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Scope, Receive, Send

# 1. Configuration for Cloud-Native Logging (stderr)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,  # Redirect logs to stderr to avoid protocol corruption
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="1.2.0", redirect_slashes=False)

# 2. Advanced Diagnostic Middleware
class DeepDiagnosticMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        
        # Capture Basic Info
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        query = scope.get("query_string", b"").decode()
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}

        log.info(f"==> [{req_id}] START: {method} {path}?{query}")
        log.info(f"==> [{req_id}] HEADERS: {json.dumps(headers, indent=2)}")

        # Wrapper to log response status
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info(f"<== [{req_id}] STATUS: {message['status']} ({elapsed:.1f}ms)")
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(DeepDiagnosticMiddleware)

# ---------------------------------------------------------------------------
# MCP Handlers
# ---------------------------------------------------------------------------
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
    
    if method == "notifications/initialized":
        log.info("Handshake Notification: Client confirmed initialization.")
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", 
            "id": msg_id, 
            "result": {"tools": [{"name": "eligibility_check", "description": "Check loan eligibility"}]}
        }
    
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/mcp")
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info(f"SSE OPEN session={session_id}")

    async def event_generator():
        try:
            # 2KB padding flushes Cloud Run GFE buffer
            yield ": " + (" " * 2048) + "\n\n"
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            while True:
                if await request.is_disconnected(): break
                await asyncio.sleep(20)
                yield ": heartbeat\n\n"
        except Exception as e:
            log.error(f"SSE Streaming Error: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform", # Crucial for Google Cloud Run
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info(f"POST BODY: {json.dumps(body, indent=2)}")
        
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        results = []
        
        for m in messages:
            res = await handle_message(m)
            if res: results.append(res)

        # Keep container alive briefly for network turns
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5)

        if not results:
            return JSONResponse({"status": "accepted"}, status_code=202)

        return JSONResponse(results if is_list else results[0])
        
    except Exception as e:
        log.error(f"POST Request Processing Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
