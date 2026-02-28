from __future__ import annotations
import json
import logging
import os
import uuid
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pega2gemini")

app = FastAPI(title="Pega2Gemini-HTTP", version="1.0.0")

# -------------------------------------------------------
# Environment
# -------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "models/gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent"

# -------------------------------------------------------
# MCP Tools Definition
# -------------------------------------------------------
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
            "properties": {
                "credit_score": {"type": "integer"},
            },
            "required": ["credit_score"],
        },
    },
]

# -------------------------------------------------------
# Gemini Call
# -------------------------------------------------------
async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY not set."

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as exc:
        log.exception("Gemini call failed")
        return f"Gemini error: {exc}"

# -------------------------------------------------------
# Tool Dispatcher
# -------------------------------------------------------
async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    log.info("Tool Call: %s | Args: %s", tool_name, arguments)

    prompt = (
        "You are an expert loan officer.\n"
        f"Tool: {tool_name}\n"
        f"Input Data: {json.dumps(arguments)}\n"
        "Provide clear professional output."
    )

    return await call_gemini(prompt)

# -------------------------------------------------------
# MCP JSON-RPC Handler
# -------------------------------------------------------
async def handle_mcp(message: dict):
    method = message.get("method")
    msg_id = message.get("id")

    # 1️⃣ Initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "Pega2Gemini",
                    "version": "1.0.0"
                },
            },
        }

    # 2️⃣ Tools List
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS
            },
        }

    # 3️⃣ Tool Call
    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        result_text = await dispatch_tool(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_text,
                    }
                ],
                "isError": False,
            },
        }

    # Default empty response
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {},
    }

# -------------------------------------------------------
# HTTP MCP Endpoint (Cloud Run Safe)
# -------------------------------------------------------
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
        log.info("MCP Request: %s", body)

        response = await handle_mcp(body)

        return JSONResponse(response)

    except Exception as e:
        log.exception("MCP processing failed")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32000,
                    "message": str(e),
                },
            },
            status_code=500,
        )

# -------------------------------------------------------
# Health Check
# -------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "service": "Pega2Gemini HTTP MCP"}
