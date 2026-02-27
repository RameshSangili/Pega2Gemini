from __future__ import annotations
import asyncio, json, logging, os, time, uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Scope, Receive, Send

app = FastAPI(title="Pega2Gemini", version="0.9.1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s")
log = logging.getLogger("pega2gemini")

GFE_FLUSH_PADDING = ": " + (" " * 2048) + "\n\n"

# --- MCP JSON-RPC Handler ---
async def handle_message(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id") # DO NOT CAST THIS. Keep it exactly as Pega sent it.
    
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
        log.info("Handshake Complete: Pega has acknowledged initialization.")
        return None 

    if method == "tools/list":
        log.info("Pega is requesting tools...")
        return {
            "jsonrpc": "2.0", 
            "id": msg_id, 
            "result": {"tools": TOOLS}
        }
    
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# --- SSE Endpoint (GET /mcp) ---
@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    async def event_generator():
        try:
            yield GFE_FLUSH_PADDING 
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"
            while True:
                if await request.is_disconnected(): break
                await asyncio.sleep(20)
                yield ": heartbeat\n\n"
        except Exception as e: log.error(f"SSE Error: {e}")
    return StreamingResponse(event_generator(), media_type="text/event-stream", 
                             headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"})

# --- Message Endpoint (POST /mcp/messages) ---
@app.post("/mcp/messages")
async def mcp_message(request: Request):
    try:
        body = await request.json()
        # Handle Pega sending a single object instead of a list
        is_list = isinstance(body, list)
        messages = body if is_list else [body]
        
        responses = []
        for msg in messages:
            method = msg.get("method")
            log.info(f"IN ==> Method: {method} ID: {msg.get('id')}")
            
            resp = await handle_message(msg)
            if resp:
                responses.append(resp)

        # Logic to return exactly what Pega expects:
        # If Pega sent a list, return a list. If Pega sent an object, return an object.
        if not responses:
            return JSONResponse({"status": "accepted"}, status_code=202)
        
        final_resp = responses if is_list else responses[0]
        
        # Keep container alive briefly for 'initialize'
        if any(m.get("method") == "initialize" for m in messages):
            await asyncio.sleep(0.5)
            
        return JSONResponse(final_resp)
    except Exception as e:
        log.error(f"POST Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

# --- Config & Tools ---
TOOLS = [
    {
        "name": "eligibility_check", 
        "description": "Check loan eligibility", 
        "inputSchema": {
            "type": "object", 
            "properties": {
                "income": {"type": "number", "description": "Monthly income"}
            }, 
            "required": ["income"]
        }
    }
]

@app.get("/health")
async def health(): return {"status": "ok"}
