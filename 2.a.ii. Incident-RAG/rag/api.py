"""FastAPI service for the incident RAG.

Run:  uvicorn rag.api:app --reload
POST /query {"question": "..."}  ->  {answer, citations, contexts}
"""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import answer

load_dotenv()
app = FastAPI(title="Incident-RAG")


class QueryRequest(BaseModel):
    question: str
    k: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    return answer(req.question, k=req.k)
