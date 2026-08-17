"""FastAPI wrapper around the DevOps Copilot agent.

Run:  uvicorn app.api:app --reload
Then: POST /chat  {"message": "why did pipe-1001 fail?", "thread_id": "abc"}

Note on write actions: in the API there is no interactive terminal, so by default
create_ticket is NOT auto-executed — the agent returns the proposed ticket. Set
env AUTO_CONFIRM_WRITES=1 to auto-approve (use only in trusted/demo contexts), or
build a proper two-step approve endpoint for production.

Embedding as a widget in another UI: this endpoint is meant to be called directly
from a browser-based chat widget in a different app. Set CORS_ALLOWED_ORIGINS to a
comma-separated list of the host app's origin(s) (defaults to "*" for local demo
use — lock this down before exposing the API publicly). The widget should generate
its own thread_id (e.g. a UUID stored in localStorage) and reuse it across messages
to keep conversation continuity, the same way streamlit_app.py does per session.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from . import tools
from .agent import build_agent

load_dotenv()

if os.environ.get("AUTO_CONFIRM_WRITES") == "1":
    tools.CONFIRM_CALLBACK = lambda proposal: True  # noqa: E731 (demo only)

app = FastAPI(title="Agentic DevOps Copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
_agent = build_agent()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread = {"configurable": {"thread_id": req.thread_id}}
    result = _agent.invoke({"messages": [HumanMessage(content=req.message)]}, thread)
    return ChatResponse(reply=result["messages"][-1].content)
