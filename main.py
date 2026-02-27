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

# 1. Initialize FastAPI (Disable auto-redirects to stop 307 loops)
app = FastAPI(title="Pega2Gemini", version="1.5.0", redirect_slashes=False)

# 2. Cloud-Native Logging (stderr ensures logs appear in Cloud Run console)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding is mandatory to force Google Front End (GFE) to flush the buffer
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
        log.info("==> Handshake Finalized by Pega")
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
                        "description": "Check loan eligibility based on profile",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "income": {"type": "number", "description": "Monthly income"},
                                "credit_score": {"type": "integer", "description": "Credit score"}
                            },
                            "required": ["income", "credit_score"]
                        }
                    }
                ]
            }
        }
    
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/mcp")
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info(f"SSE OPEN session={session_id}")

    async def event_generator():
        try:
            # Step 1: Force GFE Flush with 2KB padding
            yield GFE_FLUSH_PADDING
            
            # Step 2: Send endpoint with absolute relative path
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info(f"SSE [{session_id}] endpoint sent")
            
            # Step 3: Heartbeat Loop (Keeps Cloud Run socket active)
            while True:
                if await request.is_disconnected():
                    log.info(f"SSE [{session_id}] client disconnected")
                    break
                await asyncio.sleep(15)
                yield ": heartbeat\n\n"
        except Exception as exc:
            log.error(f"SSE Stream Error: {exc}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform", # Vital for Cloud Run
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff", # Prevents GFE buffering
        },
    )

@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info(f"IN ==> Method: {body.get('method')} ID: {body.get('id')}")
        
        # Determine if Pega sent a single object or a list
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        
        results = []
        for m in messages:
            res = await handle_message(m)
            if res: results.append(res)

        # Keep container alive briefly for 'initialize' turn-around
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.8)

        if not results:
            return JSONResponse({"status": "accepted"}, status_code=202)

        # CRITICAL: Return Object for Object, List for List to satisfy Apache-CXF
        return JSONResponse(results if is_list else results[0])
        
    except Exception as e:
        log.error(f"POST Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
