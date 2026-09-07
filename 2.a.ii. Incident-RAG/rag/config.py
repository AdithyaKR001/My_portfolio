"""Central configuration, all overridable via environment variables.

Local-first defaults: Ollama for both the chat model and embeddings, Chroma as a
local persistent vector store. No API keys required.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))

# LLM (chat) — Ollama by default, OpenAI optional
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Embeddings
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama").lower()   # ollama | openai
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Retrieval / chunking
COLLECTION = os.environ.get("CHROMA_COLLECTION", "incidents")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
TOP_K = int(os.environ.get("TOP_K", "4"))
RERANK = os.environ.get("RERANK", "1") == "1"   # lightweight lexical rerank on/off
