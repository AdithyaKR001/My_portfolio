import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.backends import get_backend
from app.batcher import DynamicBatcher
from app.cache import EmbeddingCache
from app.config import settings
from app.metrics import embed_requests_total, http_request_duration_seconds

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = get_backend()
    batcher = DynamicBatcher(
        backend=backend,
        max_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )
    cache = EmbeddingCache(
        max_size=settings.cache_max_size,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    batcher.start()
    state["backend"] = backend
    state["batcher"] = batcher
    state["cache"] = cache
    yield
    await batcher.stop()
    state.clear()


app = FastAPI(title="Model Serving at Scale (mini)", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    cache_hits: int
    cache_misses: int


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": settings.backend}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    start = time.perf_counter()
    cache: EmbeddingCache = state["cache"]
    batcher: DynamicBatcher = state["batcher"]

    results: list[list[float] | None] = [None] * len(req.texts)
    pending: list[tuple[int, str]] = []
    cache_hits = 0

    for i, text in enumerate(req.texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
            cache_hits += 1
            embed_requests_total.labels(cache="hit").inc()
        else:
            pending.append((i, text))
            embed_requests_total.labels(cache="miss").inc()

    if pending:
        embeddings = await asyncio.gather(*(batcher.submit(text) for _, text in pending))
        for (i, text), embedding in zip(pending, embeddings):
            results[i] = embedding
            cache.set(text, embedding)

    http_request_duration_seconds.labels(endpoint="/embed").observe(time.perf_counter() - start)

    return EmbedResponse(
        embeddings=results,  # type: ignore[arg-type]
        cache_hits=cache_hits,
        cache_misses=len(pending),
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
