import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
app = FastAPI()


def build_system_prompt(level: str, topic: str) -> str:
    return (
        "You are Study Partner, a friendly, patient tutor.\n"
        f"The student's education level: {level}.\n"
        f"The topic or unit they want to study: {topic}.\n\n"
        "Rules:\n"
        "- Match your vocabulary and depth to the student's level.\n"
        "- Teach one small idea at a time, with a simple example.\n"
        "- After explaining, ask one short question to check understanding.\n"
        "- If the student is wrong, explain kindly and try a different angle.\n"
        "- Stay on the chosen topic; if asked something unrelated, gently steer back.\n"
        "- If you are not sure about a fact, say so instead of guessing."
    )


class ChatRequest(BaseModel):
    message: str = Field(max_length=2000)
    level: str = Field(max_length=100)
    topic: str = Field(max_length=200)
    history: list[dict] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Keep only valid, recent messages (controls cost and bad input)
    history = [
        m for m in req.history[-20:]
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    messages = [{"role": "system", "content": build_system_prompt(req.level, req.topic)}]
    messages += history
    messages.append({"role": "user", "content": req.message})
    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# Must stay last: serves the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")