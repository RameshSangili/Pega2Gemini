from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import os
import uuid
import httpx

app = FastAPI(title="Pega2Gemini", version="0.1.0")

# ---------------------------------------------------------------------------
# Configuration – set GEMINI_API_KEY as an environment variable in Render
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ---------------------------------------------------------------------------
# Per-session queues { session_id -> asyncio.Queue }
# Each SSE connection gets its own queue so responses go to the right client
# ---------------------------------------------------------------------------
_sessions: dict = {}

# ---------------------------------------------------------------------------
# Tool definitions (MCP inputSchema format)
# ---------------------------------------------------------------------------
TOOLS: list = [
    {
        "name": "eligibility_check",
        "description": (
            "Check whether an applicant is eligible for a loan based on "
            "income, credit score, employment status and requested amount."
        ),
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
        "description": (
            "Recommend the best loan product for the applicant's profile "
            "and explain interest rate range, tenure options and caveats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "Purpose of the loan"},
                "amount": {"type": "number", "description": "Loan amount in USD"},
                "credit_score": {"type": "integer", "description": "Applicant credit score"},
                "tenure_months": {"type": "integer", "description": "Desired repayment period in months"},
            },
            "required": ["purpose", "amount", "credit_score"],
        },
    },
    {
        "name": "credit_summary",
        "description": (
            "Summarise an applicant's credit profile and highlight key "
            "risk factors that may affect loan approval."
        ),
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
# Gemini helper
# ---------------------------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY environment variable is not set on Render."
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        return f"Gemini call failed: {exc}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    args_json = json.dumps(arguments, indent=2)
    prompts = {
        "eligibility_check": (
            "You are a loan underwriting assistant. "
            "Given the following applicant details decide whether they are eligible for a loan "
            "and explain why.\n\nApplicant details:\n"
            + args_json
            + "\n\nProvide a clear YES or NO decision followed by a brief explanation."
        ),
        "loan_recommendation": (
            "You are a loan advisor. Recommend the best loan product for this applicant "
            "and explain interest rate range, tenure options, and any caveats.\n\nApplicant:\n"
            + args_json
        ),
        "credit_summary": (
            "You are a credit analyst. Summarise the credit profile below, highlight the top "
            "3 risk factors, and suggest improvements.\n\nCredit details:\n"
            + args_json
        ),
    }
    prompt = prompts.get(tool_name)
    if not prompt:
        return f"Unknown tool: {tool_name}"
    return await call_gemini(prompt)


# ---------------------------------------------------------------------------
# MCP JSON-RPC handler – returns response dict or None for notifications
# ---------------------------------------------------------------------------
async def handle_mcp_message(message: dict):
    method: str = message.get("method", "")
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "0.1.0"},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result_text = await dispatch_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return None


# ---------------------------------------------------------------------------
# GET /mcp/ – SSE stream
#
# KEY FIX: We immediately send the MCP handshake (initialize + tools/list)
# as soon as the client connects, WITHOUT waiting for POSTed requests.
# This is what Pega actually expects – it opens the SSE stream and waits
# for the server to announce itself, then sends tool-call requests via POST.
# ---------------------------------------------------------------------------
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_generator():
        try:
            # ── Step 1: tell Pega where to POST requests ─────────────────
            yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"

            # ── Step 2: immediately send initialize response ──────────────
            # Pega's SSE MCP client expects the server to proactively send
            # its initialize result right after the endpoint event.
            init_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "Pega2Gemini", "version": "0.1.0"},
                },
            }
            yield f"data: {json.dumps(init_response)}\n\n"

            # ── Step 3: immediately send tools list ───────────────────────
            tools_response = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": TOOLS},
            }
            yield f"data: {json.dumps(tools_response)}\n\n"

            # ── Step 4: keep stream alive, drain queue for tool responses ─
            while not await request.is_disconnected():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /mcp/messages/ – Pega POSTs JSON-RPC requests here
# ---------------------------------------------------------------------------
@app.post("/mcp/messages/")
async def mcp_messages(request: Request, session_id: str = ""):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    # Find the right session queue (fall back to first available)
    queue = _sessions.get(session_id)
    if queue is None and _sessions:
        queue = next(iter(_sessions.values()))

    if queue is None:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32000, "message": "No active SSE session"}},
            status_code=503,
        )

    messages = body if isinstance(body, list) else [body]
    for msg in messages:
        response = await handle_mcp_message(msg)
        if response is not None:
            await queue.put(response)

    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "sessions": len(_sessions)}
