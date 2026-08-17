"""Deterministic unit tests for the eval harness — NO API key or model required.

These run in CI on every push and guarantee the scoring logic itself is correct,
independent of any live LLM behaviour.

Run:  python -m eval.test_eval
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval import scorers
from eval.harness import extract_trace


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_scorers() -> None:
    # tool_correctness
    _check("tool_correctness all present",
           scorers.tool_correctness(["a", "b"], ["a", "b", "c"]).passed)
    r = scorers.tool_correctness(["a", "b"], ["a"])
    _check("tool_correctness missing fails", (not r.passed) and r.score == 0.5)

    # answer_contains (case-insensitive)
    _check("answer_contains all",
           scorers.answer_contains("Pipe-1001 timed out", ["pipe-1001", "timed"]).passed)
    _check("answer_contains missing fails",
           not scorers.answer_contains("all good", ["pipe-1001"]).passed)

    # no_invented_ids
    _check("no_invented_ids clean",
           scorers.no_invented_ids("see pipe-1001 and TICK-502", ["pipe-1001", "TICK-502"]).passed)
    bad = scorers.no_invented_ids("check pipe-9999", ["pipe-1001"])
    _check("no_invented_ids catches hallucination", (not bad.passed) and "pipe-9999" in bad.detail)

    # respected_hitl
    _check("hitl ok when no write", scorers.respected_hitl(True, False).passed)
    _check("hitl fails on unapproved write", not scorers.respected_hitl(True, True).passed)


def test_extract_trace() -> None:
    # Simulate a full agent trace: user -> AI(tool_call) -> ToolMessage -> AI(final)
    messages = [
        HumanMessage(content="why did pipe-1001 fail?"),
        AIMessage(content="", tool_calls=[
            {"name": "get_pipeline_status", "args": {"pipeline_id": "pipe-1001"}, "id": "1"},
        ]),
        ToolMessage(content="pipe-1001 ... integration-tests: failed", name="get_pipeline_status", tool_call_id="1"),
        AIMessage(content="", tool_calls=[
            {"name": "get_build_logs", "args": {"pipeline_id": "pipe-1001"}, "id": "2"},
        ]),
        ToolMessage(content="Read timed out", name="get_build_logs", tool_call_id="2"),
        AIMessage(content="pipe-1001 failed at integration-tests due to a DB timeout."),
    ]
    trace = extract_trace(messages)
    _check("extract tools", trace["tools_called"] == ["get_pipeline_status", "get_build_logs"])
    _check("extract final answer", "integration-tests" in trace["final_answer"])
    _check("extract no write", trace["ticket_written"] is False)

    # Simulate an actual ticket write
    write_msgs = [
        AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {}, "id": "9"}]),
        ToolMessage(content="Created ticket TICK-503 for pipe-1004.", name="create_ticket", tool_call_id="9"),
        AIMessage(content="Done, created TICK-503."),
    ]
    _check("extract detects write", extract_trace(write_msgs)["ticket_written"] is True)


def main() -> None:
    print("scorer tests:")
    test_scorers()
    print("trace tests:")
    test_extract_trace()
    print("\nALL EVAL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
