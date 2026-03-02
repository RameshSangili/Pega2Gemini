from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MCP")

# ============================================================
# APP
# ============================================================
app = FastAPI(title="Pega MCP Multi-Agent Router", version="3.0.1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# sessions[session_id] = {"queue": asyncio.Queue[str], "created": float, "last_seen": float}
_sessions: Dict[str, Dict[str, Any]] = {}

PING_SECONDS = 10 # keep-alive for Pega/Cloud Run


# ============================================================
# TOOLS
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
# GEMINI
# ============================================================
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        # IMPORTANT: for Pega, return a helpful message, do not crash
        return "GEMINI_API_KEY not configured in Cloud Run environment."

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
            # If Gemini fails, we want to log and return readable error (Pega can display it)
            if resp.status_code >= 400:
                log.error("Gemini HTTP %s body=%s", resp.status_code, resp.text[:800])
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
# JSON-RPC RESP HELPERS (put JSON string into SSE queue)
# ============================================================
async def push_jsonrpc(session_id: str, payload: dict) -> None:
    msg = json.dumps(payload, ensure_ascii=False)
    await _sessions[session_id]["queue"].put(msg)
    log.info("SSE OUT session=%s payload=%s", session_id, msg)


def jsonrpc_error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ============================================================
# MCP HANDLER
# ============================================================
async def handle_message(session_id: str, message: dict) -> None:
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    log.info("MCP IN session=%s method=%s id=%s body=%s", session_id, method, msg_id, message)

    try:
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "PegaMCPRouter", "version": "3.0.1"},
                },
            }
            await push_jsonrpc(session_id, resp)
            return

        # Pega sometimes sends this. It is a notification: no response required.
        if method == "notifications/initialized":
            log.info("notifications/initialized received session=%s", session_id)
            return

        if method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
            await push_jsonrpc(session_id, resp)
            return

        if method == "tools/call":
            tool_name = (params.get("name") or "").strip()
            args = params.get("arguments") or {}

            result_text = await route_tool(tool_name, args)

            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },
            }
            await push_jsonrpc(session_id, resp)
            return

        # Unknown method -> JSON-RPC error (better for Pega than empty result)
        await push_jsonrpc(session_id, jsonrpc_error(msg_id, -32601, f"Method not supported: {method}"))

    except Exception as e:
        log.exception("handle_message failed")
        await push_jsonrpc(session_id, jsonrpc_error(msg_id, -32000, f"Server error: {e}"))


# ============================================================
# SSE ENDPOINT (PEGA REQUIRED)
# ============================================================
@app.get("/mcp")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "queue": asyncio.Queue(),
        "created": time.time(),
        "last_seen": time.time(),
    }

    log.info("SSE START session=%s", session_id)

    async def event_stream():
        try:
            # REQUIRED FIRST EVENT: endpoint
            yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"

            while True:
                if await request.is_disconnected():
                    log.info("SSE DISCONNECT session=%s", session_id)
                    break

                try:
                    msg = await asyncio.wait_for(_sessions[session_id]["queue"].get(), timeout=PING_SECONDS)
                    # IMPORTANT: Pega expects JSON-RPC as plain data line
                    yield f"data: {msg}\n\n"
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
# MESSAGE ENDPOINT
# ============================================================
@app.post("/mcp/messages")
async def mcp_message(request: Request, session_id: str):
    if session_id not in _sessions:
        log.error("Invalid session_id=%s", session_id)
        return JSONResponse({"error": "Invalid session"}, status_code=404)

    body = await request.json()
    _sessions[session_id]["last_seen"] = time.time()

    # Return quickly — do work async
    asyncio.create_task(handle_message(session_id, body))
    return JSONResponse({"status": "accepted"})


# ============================================================
# HEALTH (optional but nice)
# ============================================================
@app.get("/")
def health():
    return {"status": "ok", "name": "PegaMCPRouter", "version": "3.0.1"}
