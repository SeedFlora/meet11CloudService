"""HTTP API dan chat UI lokal untuk RAG."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from psycopg import Error as DatabaseError

from rag import answer_question, readiness

app = FastAPI(title="Cloud Notes RAG", version="0.11")


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=1000)


class AskInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "cloud-notes-rag"}


@app.get("/ready")
def ready():
    try:
        return readiness()
    except (RuntimeError, OSError, DatabaseError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ask")
def ask(payload: AskInput):
    try:
        return answer_question(payload.question, payload.top_k, [turn.model_dump() for turn in payload.history])
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
