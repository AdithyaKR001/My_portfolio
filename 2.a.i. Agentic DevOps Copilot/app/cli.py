"""Local CLI chat loop for the DevOps Copilot.

Run:  python -m app.cli
Type 'exit' to quit. Human-in-the-loop: when the agent proposes creating a
ticket, you'll be asked to confirm on the terminal before it's written.
"""
from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from . import tools
from .agent import build_agent


def terminal_confirm(proposal: str) -> bool:
    print("\n  [human-in-the-loop] The agent wants to perform a write action:")
    print("  " + proposal.replace("\n", "\n  "))
    answer = input("  Approve? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def main() -> None:
    load_dotenv()
    tools.CONFIRM_CALLBACK = terminal_confirm  # wire up HITL for the terminal
    agent = build_agent()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("DevOps Copilot — ask about pipelines, failures, resources, tickets.")
    print("Examples:")
    print("  - Which pipelines are failing right now?")
    print("  - Why did pipe-1001 fail?")
    print("  - How much did pipe-1005 cost to run?")
    print("  - Open a ticket for the pipe-1004 build failure.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in ("exit", "quit"):
            break
        result = agent.invoke({"messages": [HumanMessage(content=user)]}, thread)
        print("copilot >", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
