from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict

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

app = FastAPI(title="Pega MCP Multi-Agent Router", version="3.0.0")

PORT = int(os.environ.get("PORT", 8080))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Session storage
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
            "properties": {
                "profile": {"type": "string"},
            },
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
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.error("Gemini error: %s", e)
        return f"Gemini error: {e}"


async def route_tool(name: str, args: dict) -> str:
    log.info("Routing tool: %s", name)

    if name == "loan_analysis":
        return await call_gemini(
            f"Analyze this loan application: {json.dumps(args)}"
        )

    if name == "risk_evaluation":
        return await call_gemini(
            f"Evaluate financial risk profile: {json.dumps(args)}"
        )

    return "Unknown tool."

# ============================================================
# MCP JSON-RPC HANDLER
# ============================================================

async def build_response(message: dict):
    method = message.get("method")
    msg_id = message.get("id")

    log.info("MCP Method: %s", method)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "PegaMCPRouter",
                    "version": "3.0.0",
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = message.get("params", {})
        result_text = await route_tool(
            params.get("name"), params.get("arguments", {})
        )

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }

    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

# ============================================================
# SSE ENDPOINT (PEGA REQUIRED)
# ============================================================

@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    _sessions[session_id] = queue

    log.info("SSE START session=%s", session_id)

    async def event_stream():
        try:
            # REQUIRED FIRST EVENT
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"

            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected session=%s", session_id)
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"

        finally:
            _sessions.pop(session_id, None)
            log.info("SSE END session=%s", session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

# ============================================================
# MESSAGE HANDLER
# ============================================================

@app.post("/mcp/messages")
async def mcp_message(request: Request, session_id: str):
    body = await request.json()
    log.info("MCP Request Body: %s", body)

    queue = _sessions.get(session_id)
    if not queue:
        log.error("Invalid session_id=%s", session_id)
        return JSONResponse({"error": "Invalid session"}, status_code=404)

    response = await build_response(body)
    await queue.put(response)

    return JSONResponse({"status": "accepted"})
