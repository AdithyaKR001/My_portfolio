"""The RAG chain: retrieve -> build grounded prompt -> generate answer with citations.

The prompt forces grounding: the model must answer only from the retrieved context
and must say it doesn't know if the answer isn't there. That is what makes the
faithfulness metric meaningful.
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .models import build_llm
from .retriever import retrieve

SYSTEM = """You are an incident-resolution assistant for a software engineering org.
Answer the engineer's question using ONLY the retrieved context below. Rules:
- Base every statement on the context. If the answer is not in the context, say
  "I don't have that in the knowledge base." Do not use outside knowledge.
- Cite the document ids you used in square brackets, e.g. [INC-1042].
- Be concise: give the likely root cause and the concrete fix.
"""


def format_context(hits: List[Dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h['doc_id']}] {h['title']}\n{h['content']}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, k: int | None = None) -> Dict:
    hits = retrieve(question, k=k)
    context = format_context(hits)
    llm = build_llm()
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    resp = llm.invoke(messages)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "question": question,
        "answer": text,
        "citations": [h["doc_id"] for h in hits],
        "contexts": hits,
    }
