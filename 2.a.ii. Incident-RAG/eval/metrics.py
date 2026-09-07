"""Pure, LLM-free RAG metrics — the measurement layer.

Retrieval metrics take a *ranked* list of retrieved doc ids and the set of gold
relevant ids. Generation metrics take the answer text. Everything here is a pure
function so it can be unit-tested with no model or API key (see eval/test_eval.py).
"""
from __future__ import annotations

import re
from typing import List

CITE_PATTERN = re.compile(r"\b(?:INC-\d+|RB-\d+)\b")
_WORD = re.compile(r"[a-z0-9]+")


# ---------------- retrieval metrics ----------------

def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    topk = retrieved[:k]
    return 1.0 if any(r in topk for r in relevant) else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of the relevant docs that appear in the top-k."""
    if not relevant:
        return 1.0
    topk = set(retrieved[:k])
    return sum(1 for r in relevant if r in topk) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    rel = set(relevant)
    return sum(1 for d in topk if d in rel) / k


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """1/rank of the first relevant doc (0.0 if none retrieved). Averaged => MRR."""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


# ---------------- generation metrics ----------------

def extract_citations(answer: str) -> List[str]:
    """The doc ids the answer cites, e.g. [INC-1042]."""
    return sorted(set(CITE_PATTERN.findall(answer)))


def keyword_coverage(text: str, keywords: List[str]) -> float:
    """Fraction of expected key facts present in the text (case-insensitive)."""
    if not keywords:
        return 1.0
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t) / len(keywords)


def citation_validity(answer: str, retrieved: List[str]) -> float:
    """Faithfulness proxy: every doc id the answer cites must have actually been
    retrieved (no fabricated citations). Returns 1.0 if no citations were made."""
    cited = extract_citations(answer)
    if not cited:
        return 1.0
    retrieved_set = set(retrieved)
    return sum(1 for c in cited if c in retrieved_set) / len(cited)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
