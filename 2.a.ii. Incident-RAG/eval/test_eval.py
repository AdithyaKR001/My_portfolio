"""Deterministic unit tests for the RAG metrics — NO model or API key needed.

Runs in CI on every push to guarantee the measurement layer is correct,
independent of any embedding model or LLM behaviour.

Run:  python -m eval.test_eval
"""
from __future__ import annotations

from eval import metrics


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_retrieval() -> None:
    retrieved = ["INC-2", "RB-1", "INC-9", "INC-5"]
    relevant = ["INC-9", "RB-1"]

    _check("hit_rate hit", metrics.hit_rate_at_k(retrieved, relevant, 4) == 1.0)
    _check("hit_rate miss@1", metrics.hit_rate_at_k(retrieved, relevant, 1) == 0.0)
    _check("recall@4 = 1.0", metrics.recall_at_k(retrieved, relevant, 4) == 1.0)
    _check("recall@2 = 0.5", metrics.recall_at_k(retrieved, relevant, 2) == 0.5)
    _check("precision@4 = 0.5", metrics.precision_at_k(retrieved, relevant, 4) == 0.5)
    # first relevant is RB-1 at rank 2 -> RR = 0.5
    _check("reciprocal_rank = 0.5", metrics.reciprocal_rank(retrieved, relevant) == 0.5)
    _check("reciprocal_rank none = 0", metrics.reciprocal_rank(["X"], relevant) == 0.0)


def test_generation() -> None:
    _check("extract_citations", metrics.extract_citations("fix per [INC-1042] and [RB-100]") == ["INC-1042", "RB-100"])
    _check("keyword_coverage full", metrics.keyword_coverage("increase Pool size, Serialize", ["pool", "serialize"]) == 1.0)
    _check("keyword_coverage half", metrics.keyword_coverage("increase pool size", ["pool", "serialize"]) == 0.5)

    # citation validity: cited must be retrieved
    _check("citation valid", metrics.citation_validity("see [INC-9]", ["INC-9", "RB-1"]) == 1.0)
    _check("citation fabricated", metrics.citation_validity("see [INC-999]", ["INC-9"]) == 0.0)
    _check("citation none => 1.0", metrics.citation_validity("I don't know", ["INC-9"]) == 1.0)


def test_golden_labels_consistent() -> None:
    """Sanity-check the generated golden set: every relevant id is well-formed and
    each question has expected keywords. Catches data-generation regressions."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden_qa.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    _check("golden set non-empty", len(rows) > 0)
    ok = all(
        r["relevant_doc_ids"] and r["answer_keywords"] and
        all(d.startswith(("INC-", "RB-")) for d in r["relevant_doc_ids"])
        for r in rows
    )
    _check("golden rows well-formed", ok)


def main() -> None:
    print("retrieval metric tests:")
    test_retrieval()
    print("generation metric tests:")
    test_generation()
    print("golden data tests:")
    test_golden_labels_consistent()
    print("\nALL RAG EVAL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
