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

# 1. Initialize FastAPI (Disable auto-redirects to stop the 307 loop)
app = FastAPI(title="Pega2Gemini", version="1.3.0", redirect_slashes=False)

# 2. Advanced Diagnostic Logging (stderr for Cloud Run console)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

# 2KB padding forces Google Cloud Run (GFE) to flush the buffer immediately
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
            "id": msg_id, # String "1"
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
            # Step 1: Immediate Padding (Forces GFE Flush)
            yield GFE_FLUSH_PADDING
            
            # Step 2: Send endpoint with absolute relative path
            # Absolute URL for Cloud Run helps Pega resolve correctly
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            log.info(f"SSE [{session_id}] endpoint sent")
            
            # Step 3: Keep-Alive Loop (Keeps Cloud Run GET active)
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
            "Transfer-Encoding": "chunked", # Forces GFE into stream mode
        },
    )

@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        log.info(f"IN ==> Method: {body.get('method')} ID: {body.get('id')}")
        
        # Pega sends a single object, not a list. We must return a single object.
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        
        results = []
        for m in messages:
            res = await handle_message(m)
            if res: results.append(res)

        # Keep container alive briefly for Pega's next turn
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5)

        if not results:
            return JSONResponse({"status": "accepted"}, status_code=202)

        # CRITICAL: Return Object if Pega sent Object, List if Pega sent List
        # This fixes the JSON-RPC parsing errors in Apache-CXF
        return JSONResponse(results[0] if not is_list else results)
        
    except Exception as e:
        log.error(f"POST Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok"}
