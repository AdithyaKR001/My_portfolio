"""Automated evaluation pipeline for the DevOps Copilot.

Runs every case in eval/dataset.json through the agent, scores each with the
deterministic scorers, writes eval/report.json + eval/report.md, prints a summary,
and EXITS NON-ZERO if the pass rate falls below --threshold. That non-zero exit is
what lets CI gate merges on agent quality.

Run:  python -m eval.run_eval               # needs an LLM (OPENAI_API_KEY or Ollama)
      python -m eval.run_eval --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from app import tools
from app.agent import build_agent
from eval import scorers
from eval.harness import run_case, valid_ids

EVAL_DIR = Path(__file__).resolve().parent


def score_case(case: dict, trace: dict, ids: list) -> list:
    results = [
        scorers.tool_correctness(case.get("expected_tools", []), trace["tools_called"]),
        scorers.answer_contains(trace["final_answer"], case.get("expected_substrings", [])),
        scorers.no_invented_ids(trace["final_answer"], ids),
        scorers.respected_hitl(case.get("must_not_write", False), trace["ticket_written"]),
    ]
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8, help="minimum pass rate to succeed")
    args = ap.parse_args()

    load_dotenv()
    # Eval runs headless: no human is present to approve writes, so leave the HITL
    # confirmation channel unset. create_ticket will refuse to write — which is
    # exactly the safe behaviour the 'safety' case checks for.
    tools.CONFIRM_CALLBACK = None

    dataset = json.loads((EVAL_DIR / "dataset.json").read_text())
    ids = valid_ids()
    agent = build_agent()

    rows, n_passed = [], 0
    metric_totals: dict[str, float] = {}

    for case in dataset:
        trace = run_case(agent, case)
        results = score_case(case, trace, ids)
        passed = scorers.case_passed(results)
        n_passed += int(passed)
        for r in results:
            metric_totals[r.metric] = metric_totals.get(r.metric, 0.0) + r.score
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "passed": passed,
            "latency_sec": trace["latency_sec"],
            "tools_called": trace["tools_called"],
            "scores": {r.metric: {"score": round(r.score, 3), "passed": r.passed, "detail": r.detail} for r in results},
            "final_answer": trace["final_answer"],
        })

    n = len(dataset)
    pass_rate = n_passed / n if n else 0.0
    avg_latency = round(sum(r["latency_sec"] for r in rows) / n, 3) if n else 0.0
    metric_avgs = {m: round(v / n, 3) for m, v in metric_totals.items()}

    report = {
        "summary": {
            "cases": n,
            "passed": n_passed,
            "pass_rate": round(pass_rate, 3),
            "avg_latency_sec": avg_latency,
            "metric_averages": metric_avgs,
            "threshold": args.threshold,
        },
        "cases": rows,
    }
    (EVAL_DIR / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report)

    print(f"\n=== DevOps Copilot eval ===")
    print(f"cases: {n}  passed: {n_passed}  pass_rate: {pass_rate:.0%}  avg_latency: {avg_latency}s")
    print("metric averages:", metric_avgs)
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']:<18} tools={r['tools_called']}")
    print(f"threshold: {args.threshold:.0%}")

    if pass_rate < args.threshold:
        print(f"FAILED: pass rate {pass_rate:.0%} < threshold {args.threshold:.0%}")
        return 1
    print("OK")
    return 0


def _write_markdown(report: dict) -> None:
    s = report["summary"]
    lines = [
        "# DevOps Copilot — Evaluation Report",
        "",
        f"- Cases: **{s['cases']}**  |  Passed: **{s['passed']}**  |  Pass rate: **{s['pass_rate']:.0%}**",
        f"- Avg latency: **{s['avg_latency_sec']}s**  |  Threshold: **{s['threshold']:.0%}**",
        f"- Metric averages: {s['metric_averages']}",
        "",
        "| Case | Category | Passed | Latency | Tools called |",
        "|---|---|---|---|---|",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {'✅' if r['passed'] else '❌'} | "
            f"{r['latency_sec']}s | {', '.join(r['tools_called']) or '—'} |"
        )
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
