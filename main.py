from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

TOOLS = [
    {"name": "eligibility_check"},
    {"name": "loan_recommendation"},
    {"name": "credit_summary"},
]

@app.get("/mcp/")
async def mcp_stream(request: Request):

    async def event_generator():

        # Wait for client initialize request
        while True:
            await asyncio.sleep(0.1)

            if await request.is_disconnected():
                break

            # Proper MCP initialize response
            response = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "name": "Pega2Gemini",
                    "version": "1.0",
                    "tools": TOOLS
                }
            }

            yield f"data: {json.dumps(response)}\n\n"

            await asyncio.sleep(20)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

