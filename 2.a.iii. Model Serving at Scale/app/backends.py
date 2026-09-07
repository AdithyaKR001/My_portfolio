import asyncio
import hashlib
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.config import settings


class EmbeddingBackend(ABC):
    """A backend turns a batch of texts into a batch of embedding vectors.

    Batching happens one level up (see app.batcher); a backend just needs to
    embed whatever list it's handed as a single unit of work, so that
    real backends (Ollama, a GPU server, ...) can amortize a single model
    forward pass across the whole batch.
    """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class MockBackend(EmbeddingBackend):
    """Deterministic, dependency-free embedder that simulates real inference cost.

    Same text always maps to the same vector (hash-based feature hashing), and
    latency scales as base + per_item * batch_size — sub-linearly cheaper per
    item as batch size grows, which is what makes server-side batching show up
    as a real throughput win in the load test rather than a no-op.
    """

    def __init__(
        self,
        dim: int = 64,
        base_latency_ms: float = 8.0,
        per_item_latency_ms: float = 6.0,
    ) -> None:
        self.dim = dim
        self.base_latency_ms = base_latency_ms
        self.per_item_latency_ms = per_item_latency_ms

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        latency_s = (self.base_latency_ms + self.per_item_latency_ms * len(texts)) / 1000.0
        await asyncio.sleep(latency_s)
        return [self._embed_one(t) for t in texts]


class OllamaBackend(EmbeddingBackend):
    """Calls a real local Ollama embedding model over its /api/embed endpoint,
    which accepts a list of inputs and returns one embedding per input — so a
    batch from the queue maps to exactly one HTTP call.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]


def get_backend() -> EmbeddingBackend:
    if settings.backend == "ollama":
        return OllamaBackend(settings.ollama_base_url, settings.ollama_model)
    return MockBackend(
        dim=settings.mock_embed_dim,
        base_latency_ms=settings.mock_base_latency_ms,
        per_item_latency_ms=settings.mock_per_item_latency_ms,
    )
