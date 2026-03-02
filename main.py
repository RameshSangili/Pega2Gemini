from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MCP")

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Pega MCP Multi-Agent Router", version="3.0.2")

PORT = int(os.environ.get("PORT", 8080)) # Cloud Run sets PORT=8080 normally
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

PING_SECONDS = int(os.environ.get("PING_SECONDS", "20")) # keepalive for SSE

# Session storage: session_id -> asyncio.Queue[dict]
_sessions: Dict[str, asyncio.Queue] = {}

# ============================================================
# TOOL DEFINITIONS
# ============================================================
TOOLS = [
    {
        "name": "loan_analysis",
        "description": "Analyze loan eligibility and provide recommendation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "income": {"type": "number"},
                "credit_score": {"type": "integer"},
                "loan_amount": {"type": "number"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    },
    {
        "name": "risk_evaluation",
        "description": "Evaluate financial risk and provide compliance guidance.",
        "inputSchema": {
            "type": "object",
            "properties": {"profile": {"type": "string"}},
            "required": ["profile"],
        },
    },
]

# ============================================================
# ROUTING ENGINE
# ============================================================
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not configured."

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)

            # log useful error detail instead of only raise_for_status
            if resp.status_code >= 400:
                log.error("Gemini HTTP %s response=%s", resp.status_code, resp.text[:800])
                return f"Gemini error HTTP {resp.status_code}: {resp.text[:200]}"

            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.exception("Gemini exception")
        return f"Gemini exception: {e}"


async def route_tool(name: str, args: dict) -> str:
    log.info("Routing tool=%s args=%s", name, args)

    if name == "loan_analysis":
        return await call_gemini(f"Analyze this loan application: {json.dumps(args)}")

    if name == "risk_evaluation":
        return await call_gemini(f"Evaluate financial risk profile: {json.dumps(args)}")

    return "Unknown tool."

# ============================================================
# JSON-RPC HELPERS
# ============================================================
def jsonrpc_error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

# ============================================================
# MCP JSON-RPC HANDLER
# ============================================================
async def build_response(message: dict) -> dict | None:
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    log.info("MCP Method: %s id=%s", method, msg_id)

    # 1) initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "PegaMCPRouter", "version": "3.0.2"},
            },
        }

    # 2) Pega may send notification: no response expected
    if method == "notifications/initialized":
        log.info("Received notifications/initialized (no response)")
        return None

    # 3) tools/list
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    # 4) tools/call
    if method == "tools/call":
        tool_name = (params.get("name") or "").strip()
        args = params.get("arguments") or {}
        result_text = await route_tool(tool_name, args)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}]
            },
        }

    # 5) unknown method => error (helps Pega debug)
    return jsonrpc_error(msg_id, -32601, f"Method not supported: {method}")

# ============================================================
# SSE ENDPOINT (PEGA REQUIRED)
# ============================================================
@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    log.info("SSE START session=%s", session_id)

    async def event_stream():
        try:
            # REQUIRED FIRST EVENT (Pega expects this)
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"

            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected session=%s", session_id)
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=PING_SECONDS)

                    # IMPORTANT: keep OLD working format: event: message
                    payload = json.dumps(msg, ensure_ascii=False)

                    log.info("SSE OUT session=%s payload=%s", session_id, payload)
                    yield f"event: message\ndata: {payload}\n\n"

                except asyncio.TimeoutError:
                    yield ": ping\n\n"

        finally:
            _sessions.pop(session_id, None)
            log.info("SSE END session=%s", session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ============================================================
# MESSAGE HANDLER
# ============================================================
@app.post("/mcp/messages")
async def mcp_message(request: Request, session_id: str):
    body = await request.json()
    log.info("MCP IN session=%s body=%s", session_id, body)

    queue = _sessions.get(session_id)
    if not queue:
        log.error("Invalid session_id=%s", session_id)
        return JSONResponse({"error": "Invalid session"}, status_code=404)

    response = await build_response(body)

    # notifications/initialized returns None
    if response is not None:
        await queue.put(response)

    return JSONResponse({"status": "accepted"})

# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "name": "PegaMCPRouter", "version": "3.0.2"}
