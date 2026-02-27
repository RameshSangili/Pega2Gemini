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
# Logging – Detailed timestamps for Render log streams
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="0.1.1")

# ---------------------------------------------------------------------------
# FIX: Pure ASGI Middleware
# Prevents 'RuntimeError: Unexpected message received' in SSE streams
# ---------------------------------------------------------------------------
class DiagnosticMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        log.info("==> REQ [%s] %s %s", req_id, scope["method"], scope["path"])

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info("<== RES [%s] status=%s elapsed=%.1fms", 
                         req_id, message["status"], elapsed)
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(DiagnosticMiddleware)

# ---------------------------------------------------------------------------
# Config & Tools
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
                "employment": {"type": "string"},
                "loan_amount": {"type": "number"},
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
                "purpose": {"type": "string"},
                "amount": {"type": "number"},
                "credit_score": {"type": "integer"},
            },
            "required": ["purpose", "amount", "credit_score"],
        },
    },
    {
        "name": "credit_summary",
        "description": "Summarise credit profile and risk factors.",
        "inputSchema": {
            "type": "object",
            "properties": {"credit_score": {"type": "integer"}},
            "required": ["credit_score"],
        },
    }
]

_sessions: dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# Gemini Integration (FIXED: JSON Parsing)
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
            # FIX: Properly navigate the Gemini candidate list
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        log.error("Gemini failed: %s", exc)
        return f"Gemini error: {exc}"

# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Tool Call: %s", tool_name)
    prompt = f"System: You are an expert loan officer. Tool: {tool_name}. Input: {json.dumps(arguments)}"
    return await call_gemini(prompt)

# ---------------------------------------------------------------------------
# MCP Handlers (FIXED: SSE Formatting)
# ---------------------------------------------------------------------------
async def build_response(message: dict):
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", 
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26", # FIXED: Matches Pega requirement
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1.0.0"},
            }
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    
    if method == "tools/call":
        params = message.get("params", {})
        res_text = await dispatch_tool(params.get("name"), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": res_text}], "isError": False}
        }
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_generator():
        try:
            log.info("SSE START session=%s", session_id)
            # MCP Requirement: provide message endpoint URL first
            yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"
            
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n" # Keep-alive for Render/Proxies
        finally:
            log.info("SSE END session=%s", session_id)
            _sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no", # Critical for Render/Nginx streaming
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/mcp/messages/")
async def mcp_message(request: Request, session_id: str):
    body = await request.json()
    queue = _sessions.get(session_id)
    if not queue:
        return JSONResponse({"error": "Invalid session"}, status_code=404)
    
    response = await build_response(body)
    if response:
        await queue.put(response)
    return JSONResponse({"status": "accepted"})