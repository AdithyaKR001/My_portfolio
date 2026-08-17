"""Trace extraction + per-case execution for the eval pipeline.

`extract_trace` turns the raw LangGraph message list into a normalized dict of
what the agent *did* (tools called, whether a ticket was written, final answer).
It is pure and unit-tested, so scoring never depends on live model behaviour.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def valid_ids() -> List[str]:
    """All legitimate pipeline and ticket ids from the mock data."""
    pipes = json.loads((DATA_DIR / "pipelines.json").read_text())
    tickets = json.loads((DATA_DIR / "tickets.json").read_text())
    return [p["id"] for p in pipes] + [t["id"] for t in tickets]


def _msg_type(m) -> str:
    return type(m).__name__


def extract_trace(messages) -> Dict:
    """Normalize a LangGraph message list into {tools_called, ticket_written, final_answer}."""
    tools_called: List[str] = []
    ticket_written = False

    for m in messages:
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tools_called.append(name)
        if _msg_type(m) == "ToolMessage":
            if getattr(m, "name", "") == "create_ticket" and "Created ticket" in str(getattr(m, "content", "")):
                ticket_written = True

    final_answer = ""
    for m in reversed(messages):
        if _msg_type(m) == "AIMessage":
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                final_answer = content
                break

    return {"tools_called": tools_called, "ticket_written": ticket_written, "final_answer": final_answer}


def run_case(agent, case: Dict) -> Dict:
    """Invoke the agent on one case and return the normalized trace + latency (seconds)."""
    from langchain_core.messages import HumanMessage

    thread = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    start = time.perf_counter()
    result = agent.invoke({"messages": [HumanMessage(content=case["input"])]}, thread)
    latency = time.perf_counter() - start
    trace = extract_trace(result["messages"])
    trace["latency_sec"] = round(latency, 3)
    return trace
