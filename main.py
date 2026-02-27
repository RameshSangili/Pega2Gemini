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

# 1. Disable redirect_slashes to stop the 307/502 loop seen in logs
app = FastAPI(title="Pega2Gemini", version="3.1.0", redirect_slashes=False)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding is the secret to forcing Cloud Run GFE to flush the buffer
GFE_FLUSH_PADDING = ": " + (" " * 2048) + "\n\n"

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
# SSE Endpoint (Standardized for Cloud Run)
# ---------------------------------------------------------------------------
@app.get("/mcp")
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info(f"SSE OPEN session={session_id}")

    async def event_generator():
        try:
            # Step 1: Force GFE Flush immediately
            yield GFE_FLUSH_PADDING
            
            # Step 2: Send the actual endpoint event
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info(f"SSE [{session_id}] initial endpoint sent")
            
            # Step 3: Keep-Alive Loop (Vital for Instance-based billing)
            while True:
                if await request.is_disconnected():
                    log.info(f"SSE [{session_id}] client disconnected")
                    break
                await asyncio.sleep(15)
                yield ": keep-alive\n\n"
        except Exception as exc:
            log.error(f"SSE Stream Error: {exc}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform", # Vital for GFE
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff", # Prevents buffering
        },
    )

# ---------------------------------------------------------------------------
# POST Messages
# ---------------------------------------------------------------------------
@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info(f"IN ==> Method: {body.get('method')} ID: {body.get('id')}")
        
        # Determine if Pega sent Object or List
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        
        results = []
        for m in messages:
            res = await handle_message(m)
            if res: results.append(res)

        # Brief pause for initialize handshake turn-around
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5)

        if not results:
            return JSONResponse({"status": "accepted"}, status_code=202)

        # CRITICAL: Return Object if Pega sent Object, List if Pega sent List
        final_payload = results if is_list else results[0]
        return JSONResponse(final_payload)
        
    except Exception as e:
        log.error(f"POST Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
