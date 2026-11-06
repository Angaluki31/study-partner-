import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()

SYSTEM_PROMPT = (
    "You are Study Partner, a friendly tutor that helps students study "
    "and revise. Explain clearly with simple examples and ask short "
    "questions to check understanding."
)

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

@app.post("/api/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += req.history
    messages.append({"role": "user", "content": req.message})
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # check Groq's docs for current model names
            messages=messages,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# Must stay last: serves the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")