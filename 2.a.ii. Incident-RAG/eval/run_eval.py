"""Automated RAG evaluation pipeline.

For every labelled question in data/eval/golden_qa.jsonl it measures retrieval
quality (hit-rate@k, recall@k, precision@k, MRR) and — unless --retrieval-only —
generation quality (answer key-fact coverage, citation validity / faithfulness
proxy) and latency. Writes eval/report.json (for the dashboard) + eval/report.md,
prints a summary, and EXITS NON-ZERO if the gate metric is below --threshold.

Run:  python -m eval.run_eval                 # full (retrieval + generation)
      python -m eval.run_eval --retrieval-only  # faster, embeddings only, no LLM
      python -m eval.run_eval --k 4 --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from eval import metrics
from rag import config

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "eval" / "golden_qa.jsonl"
OUT = Path(__file__).resolve().parent


def load_golden() -> list:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--threshold", type=float, default=0.8, help="min mean hit_rate@k to pass")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N (0 = all)")
    args = ap.parse_args()

    load_dotenv()
    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]

    # Import lazily so --help works without heavy deps.
    from rag.retriever import retrieve
    if not args.retrieval_only:
        from rag.pipeline import answer

    rows = []
    by_cat = defaultdict(list)
    agg = defaultdict(list)

    for case in golden:
        relevant = case["relevant_doc_ids"]
        t0 = time.perf_counter()
        if args.retrieval_only:
            hits = retrieve(case["question"], k=args.k)
            retrieved = [h["doc_id"] for h in hits]
            ans_text, citations = "", []
        else:
            res = answer(case["question"], k=args.k)
            retrieved = [h["doc_id"] for h in res["contexts"]]
            ans_text, citations = res["answer"], res["citations"]
        latency = time.perf_counter() - t0

        m = {
            "hit_rate": metrics.hit_rate_at_k(retrieved, relevant, args.k),
            "recall": metrics.recall_at_k(retrieved, relevant, args.k),
            "precision": metrics.precision_at_k(retrieved, relevant, args.k),
            "mrr": metrics.reciprocal_rank(retrieved, relevant),
        }
        if not args.retrieval_only:
            m["answer_match"] = metrics.keyword_coverage(ans_text, case["answer_keywords"])
            m["citation_validity"] = metrics.citation_validity(ans_text, retrieved)

        for key, val in m.items():
            agg[key].append(val)
            by_cat[case["category"]].append((key, val))

        rows.append({
            "id": case["id"], "category": case["category"], "question": case["question"],
            "relevant": relevant, "retrieved": retrieved,
            "metrics": {k: round(v, 3) for k, v in m.items()},
            "latency_sec": round(latency, 3),
            "answer": ans_text,
        })

    summary = {
        "cases": len(rows),
        "k": args.k,
        "mode": "retrieval-only" if args.retrieval_only else "full",
        "metrics": {k: round(metrics.mean(v), 3) for k, v in agg.items()},
        "avg_latency_sec": round(metrics.mean([r["latency_sec"] for r in rows]), 3),
        "threshold": args.threshold,
        "by_category": _by_category(by_cat),
    }
    report = {"summary": summary, "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report)

    print("\n=== Incident-RAG evaluation ===")
    print(f"cases: {summary['cases']}  mode: {summary['mode']}  k: {args.k}")
    for k, v in summary["metrics"].items():
        print(f"  {k:<18} {v:.3f}")
    print(f"  avg_latency        {summary['avg_latency_sec']}s")

    gate = summary["metrics"].get("hit_rate", 0.0)
    print(f"gate metric hit_rate@{args.k} = {gate:.3f} (threshold {args.threshold})")
    if gate < args.threshold:
        print("FAILED: below threshold")
        return 1
    print("OK")
    return 0


def _by_category(by_cat) -> dict:
    out = {}
    for cat, pairs in by_cat.items():
        buckets = defaultdict(list)
        for key, val in pairs:
            buckets[key].append(val)
        out[cat] = {k: round(metrics.mean(v), 3) for k, v in buckets.items()}
    return out


def _write_markdown(report: dict) -> None:
    s = report["summary"]
    lines = [
        "# Incident-RAG — Evaluation Report",
        "",
        f"- Cases: **{s['cases']}**  |  mode: **{s['mode']}**  |  k = **{s['k']}**",
        f"- Avg latency: **{s['avg_latency_sec']}s**  |  gate: hit_rate@{s['k']} ≥ {s['threshold']}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in s["metrics"].items():
        lines.append(f"| {k} | {v:.3f} |")
    (Path(__file__).resolve().parent / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
