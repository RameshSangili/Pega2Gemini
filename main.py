from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

TOOLS = [
    {"name": "eligibility_check"},
    {"name": "loan_recommendation"},
    {"name": "credit_summary"},
]

@app.get("/mcp/")
async def mcp_stream():

    async def event_generator():
        # 1️⃣ INIT event (required for Pega)
        yield "event: init\n"
        yield f"data: {json.dumps({'name':'Pega2Gemini MCP','version':'1.0'})}\n\n"

        await asyncio.sleep(0.1)

        # 2️⃣ TOOLS event
        yield "event: tools\n"
        yield f"data: {json.dumps({'tools': TOOLS})}\n\n"

        # 3️⃣ Keep connection alive
        while True:
            await asyncio.sleep(15)
            yield "event: keepalive\n"
            yield "data: ping\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
