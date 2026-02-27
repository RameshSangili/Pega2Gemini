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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini", version="0.2.0")

class DiagnosticMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        log.info("==> REQ [%s] %s %s", req_id, scope.get("method","?"), scope.get("path","?"))
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - t0) * 1000
                log.info("<== RES [%s] status=%s elapsed=%.1fms", req_id, message["status"], elapsed)
            await send(message)
        await self.app(scope, receive, send_wrapper)

app.add_middleware(DiagnosticMiddleware)

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

TOOLS = [
    {
        "name": "eligibility_check",
        "description": "Check loan eligibility based on applicant profile.",
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
        "description": "Recommend the best loan product.",
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
        "description": "Summarise credit profile and risk factors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credit_score": {"type": "integer", "description": "Credit score"},
                "outstanding_debt": {"type": "number", "description": "Outstanding debt in USD"},
                "payment_history": {"type": "string", "description": "good | average | poor"},
                "num_open_accounts": {"type": "integer", "description": "Number of open accounts"},
            },
            "required": ["credit_score"],
        },
    },
]

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
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        log.error("Gemini failed: %s", exc)
        return f"Gemini error: {exc}"

async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Dispatching tool=%s", tool_name)
    prompt = (
        f"You are an expert loan officer. "
        f"Tool called: {tool_name}. "
        f"Input data: {json.dumps(arguments)}. "
        f"Provide a clear, concise professional answer."
    )
    return await call_gemini(prompt)

async def build_response(message: dict):
    method = message.get("method", "")
    msg_id = message.get("id")
    log.info("MCP method=%s id=%s", method, msg_id)

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
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params", {})
        res_text = await dispatch_tool(params.get("name", ""), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": res_text}], "isError": False},
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    log.info("SSE CONNECT session=%s", session_id)

    async def event_generator():
        try:
            yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"
            log.info("SSE [%s] endpoint sent", session_id)
            t_start = time.time()
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(15)
                yield ": ping\n\n"
        except Exception as exc:
            log.error("SSE error: %s", exc)
        finally:
            log.info("SSE [%s] closed", session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.post("/mcp/messages/")
async def mcp_message(request: Request, session_id: str = ""):
    t0 = time.perf_counter()
    log.info("POST session=%s", session_id)
    try:
        body = await request.json()
    except Exception as exc:
        log.error("Parse error: %s", exc)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )
    log.info("POST body: %s", json.dumps(body)[:800])
    messages = body if isinstance(body, list) else [body]
    responses = []
    for msg in messages:
        resp = await build_response(msg)
        if resp is not None:
            responses.append(resp)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info("POST done %.1fms, %d response(s)", elapsed, len(responses))
    if len(responses) == 1:
        return JSONResponse(responses[0], status_code=200)
    if len(responses) > 1:
        return JSONResponse(responses, status_code=200)
    return JSONResponse({"status": "accepted"}, status_code=202)

@app.get("/debug")
async def debug():
    return {
        "version": "0.2.0",
        "platform": "Cloud Run (stateless)",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "tools_count": len(TOOLS),
        "tools": [t["name"] for t in TOOLS],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP"}