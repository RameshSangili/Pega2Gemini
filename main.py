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
# Tool definitions
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
            "Recommend the best loan product for the applicant profile "
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
            "Summarise an applicant credit profile and highlight key "
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
# Per-session queues { session_id -> asyncio.Queue }
# ---------------------------------------------------------------------------
_sessions: dict = {}

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
# Build MCP response for a given request dict
# ---------------------------------------------------------------------------
async def build_response(message: dict):
    method: str = message.get("method", "")
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                # KEY FIX 1: Use plain integer-parseable version string.
                # Pega throws "Exception while converting genai tokens to integer"
                # when it sees "2025-03-26" because it tries int("2025-03-26").
                # "20250326" (no dashes) parses cleanly as an integer.
                "protocolVersion": "20250326",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Pega2Gemini", "version": "1"},
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
# GET /mcp/ – SSE stream Pega listens on
#
# KEY FIX 2: Send initialize + tools/list IMMEDIATELY on connect so Pega
# gets them within its 30-second window before any POST is needed.
# Then keep the stream alive for tool-call responses via the queue.
# ---------------------------------------------------------------------------
@app.get("/mcp/")
async def mcp_sse(request: Request):
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_generator():
        try:
            # Step 1: Advertise POST endpoint with session_id
            yield f"event: endpoint\ndata: /mcp/messages/?session_id={session_id}\n\n"

            # Step 2: Immediately push initialize result
            # (Pega's SSE MCP client expects server to self-announce)
            yield "data: " + json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "20250326",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "Pega2Gemini", "version": "1"},
                },
            }) + "\n\n"

            # Step 3: Immediately push tools list
            yield "data: " + json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": TOOLS},
            }) + "\n\n"

            # Step 4: Stay alive and stream tool-call responses
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
# POST /mcp/messages/
#
# KEY FIX 3: Instead of returning 202 Accepted (async) we process the
# request SYNCHRONOUSLY and return the actual JSON-RPC response in the
# HTTP response body. This gives Pega an immediate answer without waiting
# for the SSE stream, which eliminates the timeout race condition.
# We ALSO push the response onto the SSE queue so the stream stays in sync.
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

    messages = body if isinstance(body, list) else [body]
    responses = []

    for msg in messages:
        response = await build_response(msg)
        if response is not None:
            responses.append(response)
            # Also push to SSE stream so it stays informed
            queue = _sessions.get(session_id)
            if queue is None and _sessions:
                queue = next(iter(_sessions.values()))
            if queue:
                await queue.put(response)

    # Return synchronously – Pega reads this directly, no need to wait for SSE
    if len(responses) == 1:
        return JSONResponse(responses[0], status_code=200)
    if len(responses) > 1:
        return JSONResponse(responses, status_code=200)

    # Notification only (no response needed)
    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# Health check – also useful as a keep-alive ping target
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP", "sessions": len(_sessions)}
