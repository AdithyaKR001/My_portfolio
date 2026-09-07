"""Retrieval with an optional lightweight lexical rerank.

`retrieve` returns a ranked list of hits: {doc_id, score, title, content}. Hits are
de-duplicated to one entry per source document (a doc may produce several chunks).
"""
from __future__ import annotations

import re
from typing import Dict, List

from . import config
from .models import build_embeddings

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _open_db():
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=build_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def lexical_rerank(query: str, hits: List[Dict]) -> List[Dict]:
    """Boost hits whose text shares more query terms. A cheap, dependency-free
    stand-in for a cross-encoder reranker; combines semantic score with lexical
    overlap so exact-term matches (ids, error strings) surface."""
    q = _tokens(query)
    for h in hits:
        overlap = len(q & _tokens(h["content"])) / (len(q) or 1)
        # smaller Chroma distance = closer; convert to similarity, then blend
        similarity = 1.0 / (1.0 + h["score"])
        h["rerank_score"] = round(0.7 * similarity + 0.3 * overlap, 4)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


def retrieve(query: str, k: int | None = None, fetch: int = 12) -> List[Dict]:
    k = k or config.TOP_K
    db = _open_db()
    # fetch more than k, dedupe per doc, then (optionally) rerank and cut to k
    raw = db.similarity_search_with_score(query, k=fetch)
    seen: Dict[str, Dict] = {}
    for doc, score in raw:
        did = doc.metadata.get("doc_id", "?")
        if did not in seen or score < seen[did]["score"]:
            seen[did] = {
                "doc_id": did,
                "score": float(score),
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
            }
    hits = list(seen.values())
    hits = lexical_rerank(query, hits) if config.RERANK else sorted(hits, key=lambda h: h["score"])
    return hits[:k]
