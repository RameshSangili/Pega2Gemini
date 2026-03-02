import os
import json
import uuid
import time
import asyncio
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# ==========================================================
# CONFIG
# ==========================================================
MCP_PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "PegaMCPRouter"
SERVER_VERSION = "3.0.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("MCP")

# ==========================================================
# APP + SESSION STORE
# ==========================================================
app = FastAPI()

# sessions[session_id] = {"queue": asyncio.Queue(), "created_at": <epoch>, "last_seen": <epoch>}
sessions: Dict[str, Dict[str, Any]] = {}


def now_ts() -> float:
    return time.time()


def safe_json(obj: Any) -> str:
    """Safe JSON stringify for logs."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# ==========================================================
# MCP HELPERS (JSON-RPC responses streamed over SSE)
# ==========================================================
async def send_result(session_id: str, request_id: Any, result: Any) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }
    msg = json.dumps(payload)
    await sessions[session_id]["queue"].put(msg)
    logger.info("SSE OUT session=%s id=%s result=%s", session_id, request_id, safe_json(result))


async def send_error(session_id: str, request_id: Any, message: str, code: int = -32000) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    msg = json.dumps(payload)
    await sessions[session_id]["queue"].put(msg)
    logger.error("SSE OUT session=%s id=%s error=%s", session_id, request_id, message)


# ==========================================================
# GEMINI TOOL IMPLEMENTATION
# ==========================================================
def gemini_generate(prompt: str) -> str:
    """
    Calls Google Generative Language API using API key.
    Raises Exception with status code details if call fails.
    """
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY is missing in environment")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    logger.info("Gemini request model=%s prompt_len=%s", GEMINI_MODEL, len(prompt))

    resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SECONDS)

    logger.info("Gemini status=%s", resp.status_code)

    # Log response body (safe). If you want less verbosity, comment next line.
    logger.info("Gemini raw body=%s", resp.text[:3000])

    if resp.status_code >= 400:
        raise Exception(f"Gemini error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    # Typical response parsing
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise Exception(f"Gemini returned unexpected response: {safe_json(data)[:500]}")


async def tool_loan_analysis(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example tool: loan_analysis
    """
    income = args.get("income")
    credit_score = args.get("credit_score")
    loan_amount = args.get("loan_amount")

    prompt = (
        "You are a financial assistant. Analyze loan eligibility and provide a clear recommendation.\n"
        f"Income: {income}\n"
        f"Credit score: {credit_score}\n"
        f"Loan amount: {loan_amount}\n"
        "Return concise eligibility summary and suggested next steps."
    )

    text = gemini_generate(prompt)

    return {
        "content": [
            {"type": "text", "text": text}
        ]
    }


async def tool_risk_evaluation(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example tool: risk_evaluation
    """
    context = args.get("context", "")
    prompt = (
        "You are a risk analyst. Evaluate the risk in a structured way.\n"
        f"Context:\n{context}\n"
        "Return risks, impact, likelihood, and mitigation steps."
    )
    text = gemini_generate(prompt)
    return {
        "content": [
            {"type": "text", "text": text}
        ]
    }


# Tool registry
TOOLS = [
    {
        "name": "loan_analysis",
        "description": "Analyze loan eligibility and provide recommendation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "income": {"type": "number"},
                "credit_score": {"type": "number"},
                "loan_amount": {"type": "number"}
            },
            "required": ["income", "credit_score", "loan_amount"]
        }
    },
    {
        "name": "risk_evaluation",
        "description": "Evaluate risk based on provided context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string"}
            },
            "required": ["context"]
        }
    }
]


# ==========================================================
# SSE ENDPOINT
# ==========================================================
@app.get("/mcp")
async def mcp_sse(request: Request):
    """
    SSE handshake endpoint.
    Returns an SSE stream and provides the POST endpoint via event: endpoint.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "queue": asyncio.Queue(),
        "created_at": now_ts(),
        "last_seen": now_ts()
    }

    logger.info("SSE START session=%s client=%s", session_id, request.client.host if request.client else "unknown")

    async def event_stream():
        # Tell client where to POST subsequent MCP messages
        yield f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n"

        while True:
            if await request.is_disconnected():
                logger.info("SSE CLOSED session=%s", session_id)
                sessions.pop(session_id, None)
                break

            # Keep session alive + ping client if no traffic
            try:
                msg = await asyncio.wait_for(sessions[session_id]["queue"].get(), timeout=10)
                yield f"data: {msg}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==========================================================
# MCP MESSAGE ENDPOINT
# ==========================================================
@app.post("/mcp/messages")
async def mcp_messages(session_id: str, body: Dict[str, Any]):
    """
    Accepts JSON-RPC MCP messages. Must return quickly.
    Actual responses are streamed via SSE for the session.
    """
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Invalid session"})

    sessions[session_id]["last_seen"] = now_ts()

    logger.info("MCP IN session=%s body=%s", session_id, safe_json(body))

    # Process async so we can return accepted immediately
    asyncio.create_task(handle_message(session_id, body))

    return {"status": "accepted"}


# ==========================================================
# ROUTER
# ==========================================================
async def handle_message(session_id: str, body: Dict[str, Any]) -> None:
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {}) or {}

    try:
        if method == "initialize":
            # MCP initialize handshake
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
            }
            await send_result(session_id, request_id, result)
            return

        if method == "notifications/initialized":
            # Some clients send this; respond with nothing (or you can log only)
            logger.info("Client initialized notification session=%s", session_id)
            return

        if method == "tools/list":
            result = {"tools": TOOLS}
            await send_result(session_id, request_id, result)
            return

        if method == "tools/call":
            tool_name = (params.get("name") or "").strip()
            args = params.get("arguments") or {}

            logger.info("tools/call session=%s tool=%s args=%s", session_id, tool_name, safe_json(args))

            if tool_name == "loan_analysis":
                result = await tool_loan_analysis(args)
                await send_result(session_id, request_id, result)
                return

            if tool_name == "risk_evaluation":
                result = await tool_risk_evaluation(args)
                await send_result(session_id, request_id, result)
                return

            await send_error(session_id, request_id, f"Tool not found: {tool_name}", code=-32601)
            return

        # Unsupported method
        await send_error(session_id, request_id, f"Method not supported: {method}", code=-32601)

    except Exception as e:
        logger.exception("handle_message failed session=%s id=%s", session_id, request_id)
        await send_error(session_id, request_id, str(e), code=-32000)


# ==========================================================
# HEALTH
# ==========================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocolVersion": MCP_PROTOCOL_VERSION
    }
