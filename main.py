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

# 1. Initialize FastAPI
app = FastAPI(title="Pega2Gemini", version="1.4.0", redirect_slashes=False)

# 2. Cloud-Native Logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding forces GFE to flush
GFE_FLUSH_PADDING = ": " + (" " * 2048) + "\n\n"

# ---------------------------------------------------------------------------
# MCP Logic
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
        log.info("==> Handshake Finalized")
        return None

    if method == "tools/list":
        log.info("==> Pega is fetching tools list")
        return {
            "jsonrpc": "2.0", 
            "id": msg_id, 
            "result": {
                "tools": [
                    {
                        "name": "eligibility_check",
                        "description": "Check loan eligibility",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "income": {"type": "number"},
                                "credit_score": {"type": "integer"}
                            },
                            "required": ["income", "credit_score"]
                        }
                    }
                ]
            }
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
            # Step 1: Force GFE Flush
            yield GFE_FLUSH_PADDING
            
            # Step 2: Send Endpoint
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info(f"SSE [{session_id}] endpoint sent")
            
            # Step 3: Keep-Alive Loop
            while True:
                if await request.is_disconnected():
                    log.info(f"SSE [{session_id}] disconnected")
                    break
                await asyncio.sleep(10)
                yield ": keep-alive\n\n"
        except Exception as e:
            log.error(f"SSE Error: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform", 
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff", # Prevents GFE from sniffing/buffering
        },
    )

@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info(f"IN ==> Method: {body.get('method')} ID: {body.get('id')}")
        
        # Pega sends Object, we return Object.
        res = await handle_message(body)

        if body.get("method") == "initialize":
            await asyncio.sleep(0.5)

        if res:
            return JSONResponse(res)
        return JSONResponse({"status": "accepted"}, status_code=202)
        
    except Exception as e:
        log.error(f"POST Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
