from dotenv import load_dotenv
load_dotenv()

import os, json, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import google.generativeai as genai
from logger import get_logger

logger = get_logger()

# -----------------------------
# Gemini setup
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is empty. Gemini calls will fail.")

genai.configure(api_key=GEMINI_API_KEY)

# IMPORTANT:
# Use the model name that exists in YOUR /v1beta/models list.
# You previously saw: models/gemini-pro-latest
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")

model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Loan MCP Server", version="1.0")

# -----------------------------
# Tools (POC)
# -----------------------------
def eligibility_check(data: dict) -> dict:
    income = float(data.get("income", 0))
    credit_score = int(data.get("creditScore", 0))
    return {"eligible": bool(income >= 60000 and credit_score >= 700)}

def loan_recommendation(data: dict) -> dict:
    income = float(data.get("income", 0))
    # Simple POC rule
    return {"recommendedLoan": min(income * 5, 500000)}

def credit_summary(data: dict) -> dict:
    credit_score = int(data.get("creditScore", 0))
    return {"rating": "Good" if credit_score >= 700 else "Fair"}

TOOLS = {
    "eligibility_check": eligibility_check,
    "loan_recommendation": loan_recommendation,
    "credit_summary": credit_summary,
}

# -----------------------------
# Gemini router (streams tool name)
# -----------------------------
def stream_tool_decision(prompt: str):
    system_prompt = f"""
You are an AI router for a personal-loan application.
Choose exactly ONE tool from this list:

{list(TOOLS.keys())}

Rules:
- Return ONLY the tool name (exactly as shown)
- No explanation, no punctuation, no extra words
""".strip()

    logger.info("Gemini router called")
    logger.debug(f"Router prompt: {prompt}")

    # Stream tokens from Gemini
    resp = model.generate_content(system_prompt + "\nUser: " + prompt, stream=True)
    for chunk in resp:
        text = getattr(chunk, "text", None)
        if text:
            yield text

def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"

# -----------------------------
# MCP endpoints expected by Pega Connect MCP
# Base URL to configure in Pega: http(s)://HOST:PORT/mcp
# -----------------------------
@app.get("/mcp")
def list_tools():
    return {
        "name": "Pega2Gemini MCP",
        "version": "1.0",
        "description": "Loan AI MCP Server powered by Gemini",
        "tools": [
            {
                "name": "eligibility_check",
                "description": "Check applicant loan eligibility",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "income": {"type": "number"},
                        "creditScore": {"type": "number"}
                    },
                    "required": ["income", "creditScore"]
                }
            },
            {
                "name": "loan_recommendation",
                "description": "Recommend loan products",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "income": {"type": "number"},
                        "creditScore": {"type": "number"}
                    },
                    "required": ["income", "creditScore"]
                }
            },
            {
                "name": "credit_summary",
                "description": "Summarize credit profile",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "creditScore": {"type": "number"}
                    },
                    "required": ["creditScore"]
                }
            }
        ]
    }

@app.post("/mcp/invoke")
async def invoke(request: Request):
    body = await request.json()
    logger.info("POST /mcp/invoke")
    logger.debug(f"Incoming payload: {body}")

    prompt = (body.get("prompt") or "").strip()
    input_data = body.get("input") or {}

    def event_stream():
        start_total = time.time()
        try:
            # 1) Gemini selects tool (stream)
            tool_name = ""
            for token in stream_tool_decision(prompt):
                tool_name += token
                yield sse({"type": "ai_stream", "text": token})

            tool_name = tool_name.strip()
            logger.info(f"Gemini selected tool: {tool_name}")

            if tool_name not in TOOLS:
                logger.error(f"Invalid tool from Gemini: {tool_name}")
                yield sse({"type": "error", "message": f"Invalid tool selected: {tool_name}"})
                return

            yield sse({"type": "tool_selected", "tool": tool_name})

            # 2) Execute tool
            t0 = time.time()
            result = TOOLS[tool_name](input_data)
            tool_secs = round(time.time() - t0, 3)
            logger.info(f"Tool executed: {tool_name} in {tool_secs}s")
            logger.debug(f"Tool result: {result}")

            # 3) Final
            yield sse({"type": "final", "result": result})

        except Exception as e:
            logger.exception("MCP invoke failed")
            yield sse({"type": "error", "message": str(e)})
        finally:
            total_secs = round(time.time() - start_total, 3)
            logger.info(f"Request completed in {total_secs}s")
            yield sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok", "gemini_key": bool(GEMINI_API_KEY), "model": GEMINI_MODEL_NAME}
