"""Builds the LangGraph ReAct agent for the DevOps Copilot.

LLM is configurable via env:
  LLM_PROVIDER = ollama (default) | openai
  OLLAMA_MODEL (default llama3.1), OLLAMA_BASE_URL (default http://localhost:11434)
  OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini)

Ollama runs fully locally with no API key. We use the native langchain-ollama
integration (ChatOllama), which has first-class tool/function-calling support —
important for an agent. The model you use MUST support tools (llama3.1, qwen2.5,
mistral-nemo, etc.).
"""
from __future__ import annotations

import os

from langgraph.prebuilt import create_react_agent

from .tools import ALL_TOOLS

SYSTEM_PROMPT = """You are the DevOps Copilot, an agentic assistant for a software \
engineering team. You help engineers track CI/CD pipelines, diagnose build and test \
failures, inspect resource usage, and manage tickets.

Guidelines:
- Use the tools to fetch real data before answering. Never invent pipeline ids, \
statuses, logs, or numbers.
- When diagnosing a failure, look at the pipeline status AND the build logs, then give \
a concise root-cause hypothesis and a recommended next step.
- create_ticket is a WRITE action. Always propose the ticket and get explicit human \
confirmation first; never create a ticket without being asked to and confirmed.
- For "how many tickets" / "total tickets" questions, call count_tickets rather than \
calling list_tickets multiple times and adding the results yourself.
- For questions that depend on how long a ticket has been open, or on whether a \
ticket's pipeline is CURRENTLY failing, use list_tickets's pipeline_status filter \
(and its created_at / pipeline-status fields in the output) instead of guessing or \
combining list_pipelines and list_tickets yourself — it already does that lookup.
- Be concise. Prefer short, skimmable answers with concrete ids and numbers.
"""


def build_llm():
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def build_agent():
    """Return a compiled LangGraph agent with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    llm = build_llm()
    agent = create_react_agent(
        llm,
        ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    return agent
