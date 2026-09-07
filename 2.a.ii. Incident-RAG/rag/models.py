"""LLM + embedding factory. Keeps provider selection in one place."""
from __future__ import annotations

from . import config


def build_llm():
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)


def build_embeddings():
    if config.EMBED_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL)
