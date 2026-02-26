from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import os
import httpx

app = FastAPI(title="Pega2Gemini", version="0.1.0")

# ---------------------------------------------------------------------------
# Configuration – set GEMINI_API_KEY as an environment variable in Render
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ---------------------------------------------------------------------------
# Tool definitions (MCP schema format)
# ---------------------------------------------------------------------------
TOOLS = [
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
                "credit_score": {"type": "integer", "description": "Credit score (300-850)"},
                "employment": {"type": "string", "description": "employed | self-employed | unemployed"},
                "loan_amount": {"type": "number", "description": "Requested loan amount in USD"},
            },
            "required": ["income", "credit_score", "loan_amount"],
        },
    },
    {
        "name": "loan_recommendation",
        "description": (
            "Recommend the best loan product (personal, home, auto, etc.) "
            "for the applicant's profile and explain the reasoning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "Purpose of the loan"},
                "amount": {"type": "number", "description": "Loan amount in USD"},
                "credit_score": {"type": "integer","description": "Applicant credit score"},
                "tenure_months":{"type": "integer","description": "Desired repayment period in months"},
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
# Helper – call Gemini
# ---------------------------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY environment variable is not set."

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
# Tool dispatcher – build a natural-language prompt then call Gemini
# ---------------------------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    prompts = {
        "eligibility_check": (
            "You are a loan underwriting assistant. "
            "Given the following applicant details, decide whether they are eligible for a loan "
            "and explain why.\n\nApplicant details:\n"
            + json.dumps(arguments, indent=2)
            + "\n\nProvide a clear YES or NO eligibility decision followed by a brief explanation."
        ),
        "loan_recommendation": (
            "You are a loan advisor. Recommend the best loan product for this applicant and "
            "explain the interest rate range, tenure options, and any caveats.\n\nApplicant details:\n"
            + json.dumps(arguments, indent=2)
        ),
        "credit_summary": (
            "You are a credit analyst. Summarise the credit profile below, highlight the top "
            "3 risk factors, and suggest what the applicant can do to improve their score.\n\n"
            "Credit details:\n"
            + json.dumps(arguments, indent=2)
        ),
    }

    prompt = prompts.get(tool_name)
    if not prompt:
        return f"Unknown tool: {tool_name}"

    return await call_gemini(prompt)


# ---------------------------------------------------------------------------
# MCP message handler
# ---------------------------------------------------------------------------
async def handle_mcp_message(message: dict) -> dict | None:
    """
    Process one JSON-RPC 2.0 message and return the response dict,
    or None if no response is needed (notifications).
    """
    method = message.get("method", "")
    msg_id = message.get("id") # may be None for notifications

    # ── initialize ──────────────────────────────────────────────────────────
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

    # ── initialized (notification – no response needed) ─────────────────────
    if method == "notifications/initialized":
        return None

    # ── tools/list ──────────────────────────────────────────────────────────
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    # ── tools/call ──────────────────────────────────────────────────────────
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

    # ── ping ────────────────────────────────────────────────────────────────
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # ── unknown method ───────────────────────────────────────────────────────
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return None


# ---------------------------------------------------------------------------
# SSE endpoint – GET /mcp/
# Pega sends JSON-RPC messages as query-param or request body via POST;
# for SSE transport the client POSTs to a separate endpoint and listens here.
#
# Pega's MCP connector (SSE mode) works like this:
# 1. Client opens GET /mcp/ → server keeps the SSE stream open
# 2. Client POSTs JSON-RPC requests to POST /mcp/
# 3. Server sends responses back on the open SSE stream
# ---------------------------------------------------------------------------

# Store active SSE queues keyed by session (simplified: single global queue)
_sse_queue: asyncio.Queue = asyncio.Queue()


@app.get("/mcp/")
async def mcp_sse(request: Request):
    """SSE stream – Pega connects here and listens for responses."""

    async def event_generator():
        # Send the MCP endpoint event so Pega knows where to POST
        yield "event: endpoint\ndata: /mcp/messages/\n\n"

        while not await request.is_disconnected():
            try:
                message = await asyncio.wait_for(_sse_queue.get(), timeout=15.0)
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                # Send a keep-alive comment to prevent proxy/Render timeouts
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # important for Render / nginx proxies
        },
    )


@app.post("/mcp/messages/")
async def mcp_messages(request: Request):
    """
    Pega POSTs JSON-RPC requests here.
    We process them and push responses onto the SSE queue.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    # Handle batch or single message
    messages = body if isinstance(body, list) else [body]

    for msg in messages:
        response = await handle_mcp_message(msg)
        if response is not None:
            await _sse_queue.put(response)

    return JSONResponse({"status": "accepted"}, status_code=202)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pega2Gemini MCP"}