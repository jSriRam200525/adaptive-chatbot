import os, time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TIMEOUT_SEC    = 15.0

app = FastAPI(title="Adaptive AI Chatbot", version="1.0.0")

class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[Message] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    latency_ms: int
    usage: Dict[str, Any] = {}

async def call_llm(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.7}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
        data = r.json()
        return {
            "reply": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {})
        }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    system_prompt = {"role": "system", "content": "You are a helpful chatbot."}
    messages = [system_prompt] + [m.model_dump() for m in req.history] + [
        {"role": "user", "content": req.message}
    ]
    llm = await call_llm(messages)
    return ChatResponse(
        reply=llm["reply"],
        latency_ms=int((time.time() - start) * 1000),
        usage=llm["usage"],
    )
