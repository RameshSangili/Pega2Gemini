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

# 1. Allow redirects but handle them gracefully
app = FastAPI(title="Pega2Gemini", version="1.0.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

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
                            "properties": {"income": {"type": "number"}},
                            "required": ["income"]
                        }
                    }
                ]
            }
        }
    
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# ---------------------------------------------------------------------------
# Endpoints (Routes handle both /mcp and /mcp/ to stop 307 Redirects)
# ---------------------------------------------------------------------------

@app.get("/mcp")
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info("SSE OPEN session=%s", session_id)

    async def event_generator():
        try:
            yield GFE_FLUSH_PADDING 
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            while True:
                if await request.is_disconnected(): break
                await asyncio.sleep(20)
                yield ": heartbeat\n\n"
        except Exception: pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/mcp/messages")
@app.post("/mcp/messages/")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        # FIX: Pega sends an OBJECT, so we must return an OBJECT (not a list)
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        
        results = []
        for m in messages:
            log.info("IN ==> Method: %s", m.get("method"))
            res = await handle_message(m)
            if res: results.append(res)

        # Ensure container stays up for the handshake
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5)

        if not results:
            return JSONResponse({"status": "accepted"}, status_code=202)

        # CRITICAL: Return exactly what Pega sent. 
        # If Pega sent {}, return {}. If Pega sent [{}], return [{}].
        final_payload = results if is_list else results[0]
        return JSONResponse(final_payload)
        
    except Exception as e:
        log.error("POST Error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/health")
async def health(): return {"status": "ok"}
