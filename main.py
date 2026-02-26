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
# Logging – verbose, timestamped, goes straight to Render's log stream
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="0.1.0")

# ---------------------------------------------------------------------------
# Pure ASGI Diagnostic Middleware (SSE Compatible)
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
        
        # Log basic request info
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
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
TOOLS: list = [
    {
        "name": "eligibility_check",
        "description": "Check loan eligibility based on income, credit score, employment and amount.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "income": {"type": "number", "description": "Monthly income in USD"},
                "credit_score": {"type": "integer", "description": "Credit score 300-850"},
                "employment": {"type": "string", "description": "employed | self-employed | unemployed"},
                "loan_amount": {"type": "number", "description": "Requested loan amount in USD"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    },
    {
        "name": "loan_recommendation",
        "description": "Recommend the best loan product for the applicant profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "Purpose of the loan"},
                "amount": {"type": "number", "description": "Loan amount in USD"},
                "credit_score": {"type": "integer", "description": "Applicant credit score"},
                "tenure_months": {"type": "integer", "description": "Repayment period in months"},
            },
            "required": ["purpose", "amount", "credit_score"],
        },
    },
    {
        "name": "credit_summary",
        "description": "Summarise credit profile and highlight key risk factors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credit_score": {"type": "integer", "description": "Credit score"},
                "outstanding_debt": {"type": "number", "description": "Total outstanding debt in USD"},
                "payment_history": {"type": "string", "description": "good | average | poor"},
                "num_open_accounts": {"type": "integer", "description": "Number of open credit accounts"},
            },
            "required": ["credit_score"],
        },
    },
]

# ---------------------------------------------------------------------------
# Per-session SSE queues
# ---------------------------------------------------------------------------
_sessions: dict = {}

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY is not set!")
        return "ERROR: GEMINI_API_KEY environment variable is not set."
    log.debug("Calling Gemini, prompt length=%d chars", len(prompt))
    t0 = time.time()
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
            resp.raise_for_status()
            data = resp.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"]
            log.info("Gemini responded in %.2fs, reply length=%d chars", time.time() - t0, len(result))
            return result
    except Exception as exc:
        log.error("Gemini call failed after %.2fs: %s", time.time() - t0, exc)
        return f"Gemini call failed: {exc}"

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Dispatching tool=%s args=%s", tool_name, arguments)
    args_json = json.dumps(arguments, indent=2)
    prompts = {
        "eligibility_check": (
            "You are a loan underwriting assistant. "
            "Given the following applicant details decide whether they are eligible for a loan.\n\n"
            + args_json + "\n\nProvide a clear YES or NO decision followed by a brief explanation."
        ),
        "loan_recommendation": (
            "You are a loan advisor. Recommend the best loan product for this applicant.\n\n"
            + args_json
        ),
        "credit_summary": (
            "You are a credit analyst. Summarise the credit profile and highlight top 3 risk factors.\n\n"
            + args_json
        ),
    }
    prompt = prompts.get(tool_name)
    if not prompt:
        log.warning("Unknown tool requested: %s", tool_name)
        return f"Unknown tool: {tool_name}"
    return await call_gemini(prompt)

# ---------------------------------------------------------------------------
# MCP JSON-RPC builder
# ---------------------------------------------------------------------------
async def build_response(message: dict):
    method: str = message.get("method", "")
    msg_id = message.get("id")
    log.info(">>> MCP request method=%s id=%s", method, msg_id)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "20250326",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result_text = await dispatch_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": result_text}], "isError": False},
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    return None

# ---------------------------------------------------------------------------
# MCP Endpoints
# ---------------------------------------------------------------------------
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_generator():
        try:
            log.info("SSE CONNECT session=%s", session_id)
            # Initial endpoint event per MCP spec
            yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"
            
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            log.info("SSE DISCONNECTED session=%s", session_id)
            _sessions.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp/messages/")
async def mcp_message(request: Request, session_id: str):
    body = await request.json()
    queue = _sessions.get(session_id)
    if not queue:
        return JSONResponse({"error": "Invalid session"}, status_code=404)
    
    response = await build_response(body)
    if response:
        await queue.put(response)
    return JSONResponse({"status": "ok"})
