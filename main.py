import json
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
class QuizRequest(BaseModel):
    level: str = Field(max_length=100)
    topic: str = Field(max_length=200)
    count: int = Field(default=5, ge=1, le=10)


@app.post("/api/quiz")
def quiz(req: QuizRequest):
    prompt = (
        f"Create {req.count} multiple-choice questions for a student at this level: "
        f"{req.level}. Topic: {req.topic}.\n"
        "Return ONLY valid JSON, with no extra text and no markdown fences, "
        "in exactly this shape:\n"
        '{"questions": [{"question": "...", '
        '"options": ["option 1", "option 2", "option 3", "option 4"], '
        '"answer_index": 0, "explanation": "..."}]}\n'
        "Each question must have exactly 4 options and exactly one correct answer. "
        "answer_index is the position (0 to 3) of the correct option. "
        "Vary which position is correct."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        raise HTTPException(status_code=502, detail="Could not generate a quiz. Please try again.")

    questions = [
        q for q in data.get("questions", [])
        if isinstance(q, dict)
        and q.get("question") and q.get("explanation")
        and isinstance(q.get("options"), list) and len(q["options"]) == 4
        and isinstance(q.get("answer_index"), int) and 0 <= q["answer_index"] <= 3
    ]
    if not questions:
        raise HTTPException(status_code=502, detail="Could not generate a valid quiz. Please try again.")
    return {"questions": questions}

def ask_json(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        raise HTTPException(status_code=502, detail="Could not generate content. Please try again.")


class CardsRequest(BaseModel):
    level: str = Field(max_length=100)
    topic: str = Field(max_length=200)
    count: int = Field(default=8, ge=1, le=15)


@app.post("/api/flashcards")
def flashcards(req: CardsRequest):
    prompt = (
        f"Create {req.count} flashcards for a student at this level: {req.level}. "
        f"Topic: {req.topic}.\n"
        "Return ONLY valid JSON, with no extra text and no markdown fences, "
        "in exactly this shape:\n"
        '{"cards": [{"front": "a question or term", "back": "a short, clear answer"}]}\n'
        "Keep each front and back brief. Cover the most important ideas of the topic."
    )
    data = ask_json(prompt)
    cards = [
        c for c in data.get("cards", [])
        if isinstance(c, dict)
        and isinstance(c.get("front"), str) and c["front"]
        and isinstance(c.get("back"), str) and c["back"]
    ]
    if not cards:
        raise HTTPException(status_code=502, detail="Could not generate flashcards. Please try again.")
    return {"cards": cards}

# Locally, FastAPI serves the frontend. On Vercel, the public/ folder is served by their CDN.
if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")