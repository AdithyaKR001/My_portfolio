"""Tool layer for the Agentic DevOps Copilot.

Each tool reads from local mock JSON files in ../data. Swap the DataStore
methods for real CI/CD, ITSM, and metrics API calls to go to production —
the tool signatures the agent sees stay identical.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

# Set by the runtime (CLI vs API) to control human-in-the-loop behaviour.
# In the CLI we ask on the terminal; in the API we require pre-approval.
CONFIRM_CALLBACK = None  # type: Optional[callable]


def _load(name: str):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(name: str, obj) -> None:
    with open(DATA_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@tool
def list_pipelines(status: Optional[str] = None) -> str:
    """List CI/CD pipelines with their current status.

    Args:
        status: optional filter, one of 'success', 'failed', 'running'.
    """
    pipes = _load("pipelines.json")
    if status:
        pipes = [p for p in pipes if p["status"] == status.lower()]
    if not pipes:
        return f"No pipelines found with status={status!r}."
    lines = [f"- {p['id']} ({p['name']}, branch {p['branch']}): {p['status'].upper()}" for p in pipes]
    return "Pipelines:\n" + "\n".join(lines)


@tool
def get_pipeline_status(pipeline_id: str) -> str:
    """Get detailed stage-by-stage status for one pipeline by its id (e.g. 'pipe-1001')."""
    pipes = {p["id"]: p for p in _load("pipelines.json")}
    p = pipes.get(pipeline_id)
    if not p:
        return f"No pipeline with id {pipeline_id!r}. Use list_pipelines to see valid ids."
    stages = "\n".join(
        f"    {s['name']}: {s['status']} ({s['duration_sec']}s)" for s in p["stages"]
    )
    return (
        f"{p['id']} — {p['name']} (branch {p['branch']})\n"
        f"  overall: {p['status'].upper()}, triggered_by {p['triggered_by']}, "
        f"commit {p['commit']}, last_run {p['last_run']}\n  stages:\n{stages}"
    )


@tool
def get_build_logs(pipeline_id: str) -> str:
    """Get the build/test logs for a pipeline. Useful for diagnosing failures."""
    logs = _load("logs.json")
    if pipeline_id not in logs:
        return f"No logs for {pipeline_id!r}."
    return f"Logs for {pipeline_id}:\n{logs[pipeline_id]}"


@tool
def get_resource_usage(pipeline_id: str) -> str:
    """Get CPU, memory, build minutes and estimated cost for a pipeline's last run."""
    res = _load("resources.json")
    r = res.get(pipeline_id)
    if not r:
        return f"No resource data for {pipeline_id!r}."
    return (
        f"Resource usage for {pipeline_id}: "
        f"peak CPU {r['cpu_cores_peak']} cores, peak memory {r['memory_gb_peak']} GB, "
        f"{r['build_minutes']} build-minutes, est. cost ${r['est_cost_usd']}."
    )


def _age_str(created_at: str) -> str:
    """Elapsed time from created_at to now, e.g. '4d 6h'. Computed here so the
    agent reports a real duration instead of doing date arithmetic itself."""
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - created
    days, hours = delta.days, delta.seconds // 3600
    return f"{days}d {hours}h"


@tool
def list_tickets(status: Optional[str] = None, pipeline_status: Optional[str] = None) -> str:
    """List existing engineering tickets, optionally filtered by ticket status
    ('open', 'in_progress', 'closed') and/or by the CURRENT status of the pipeline
    each ticket is linked to ('success', 'failed', 'running'). Use pipeline_status
    for any question that needs to cross-reference a ticket against how its
    pipeline is doing right now (e.g. "tickets for pipelines that are still
    failing") — this tool does that lookup for you, so don't try to combine
    list_pipelines and list_tickets results yourself. Results include each
    ticket's creation date, how long it's been open (open_for — use this
    directly for any "longest"/"oldest"/processing-time question instead of
    computing it yourself from created_at), and its pipeline's current status.
    Note a ticket's own status is one of open/in_progress/closed — it is never
    'failed'; only a pipeline can be 'failed'. Ordered oldest-created first, so
    the first row for any filter is also the longest-standing one."""
    tickets = _load("tickets.json")
    pipeline_statuses = {p["id"]: p["status"] for p in _load("pipelines.json")}

    if status:
        tickets = [t for t in tickets if t["status"] == status.lower()]
    if pipeline_status:
        tickets = [t for t in tickets if pipeline_statuses.get(t["pipeline_id"]) == pipeline_status.lower()]
    if not tickets:
        return f"No tickets found with status={status!r}, pipeline_status={pipeline_status!r}."

    tickets = sorted(tickets, key=lambda t: t["created_at"])
    return "Tickets (oldest created first):\n" + "\n".join(
        f"- {t['id']} [{t['status']}] {t['title']} (pipeline {t['pipeline_id']} is currently "
        f"{pipeline_statuses.get(t['pipeline_id'], 'unknown').upper()}, assignee {t['assignee']}, "
        f"created {t['created_at']}, open_for {_age_str(t['created_at'])})"
        for t in tickets
    )


@tool
def count_tickets() -> str:
    """Get an accurate total ticket count broken down by status. Always use this
    tool (instead of list_tickets + manual counting) for any 'how many tickets'
    or 'total tickets' question — it computes the count directly rather than
    relying on you to sum filtered lists, which is error-prone."""
    tickets = _load("tickets.json")
    if not tickets:
        return "No tickets found. Total: 0."
    counts: dict[str, int] = {}
    for t in tickets:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    breakdown = ", ".join(f"{status}: {n}" for status, n in sorted(counts.items()))
    return f"Total tickets: {len(tickets)} ({breakdown})."


@tool
def create_ticket(title: str, description: str, pipeline_id: str) -> str:
    """Create a new engineering ticket for a failing pipeline. This is a WRITE action:
    it must be confirmed by a human before it takes effect."""
    proposal = f"CREATE TICKET -> title={title!r}, pipeline={pipeline_id!r}\n  description: {description}"
    if CONFIRM_CALLBACK is not None:
        approved = CONFIRM_CALLBACK(proposal)
        if not approved:
            return "Ticket creation was declined by the human reviewer. No ticket created."
    else:
        # No confirmation channel wired up -> refuse rather than write silently.
        return (
            "Write action requires human confirmation, but no confirmation channel is "
            "configured. Returning the proposed ticket instead of creating it:\n" + proposal
        )

    tickets = _load("tickets.json")
    new_id = f"TICK-{500 + len(tickets) + 1}"
    tickets.append({
        "id": new_id,
        "title": title,
        "status": "open",
        "pipeline_id": pipeline_id,
        "assignee": "unassigned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": description,
    })
    _save("tickets.json", tickets)
    return f"Created ticket {new_id} for {pipeline_id}."


ALL_TOOLS = [
    list_pipelines,
    get_pipeline_status,
    get_build_logs,
    get_resource_usage,
    list_tickets,
    count_tickets,
    create_ticket,
]
