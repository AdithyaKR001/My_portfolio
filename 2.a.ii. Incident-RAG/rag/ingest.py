"""Ingestion: load the corpus -> chunk -> embed -> persist into Chroma.

Run:  python -m rag.ingest
Re-running rebuilds the collection from scratch (idempotent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .models import build_embeddings


def load_corpus() -> List[Document]:
    docs: List[Document] = []
    for fname in ("incidents.jsonl", "runbooks.jsonl"):
        path = config.CORPUS_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(Document(
                page_content=rec["body"],
                metadata={
                    "doc_id": rec["id"],
                    "type": rec["type"],
                    "category": rec.get("category", ""),
                    "service": rec.get("service", ""),
                    "title": rec.get("title", ""),
                },
            ))
    return docs


def build_index() -> int:
    """(Re)build the Chroma collection. Returns the number of chunks indexed."""
    from langchain_chroma import Chroma

    docs = load_corpus()
    if not docs:
        raise SystemExit("No corpus found. Run: python scripts/generate_data.py")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma(
        collection_name=config.COLLECTION,
        embedding_function=build_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    # Rebuild cleanly so re-ingest is idempotent.
    try:
        vectordb.reset_collection()
    except Exception:
        pass
    vectordb.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into Chroma collection '{config.COLLECTION}' at {config.CHROMA_DIR}")
