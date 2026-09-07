"""Deterministic unit tests for caching, the mock backend, and dynamic batching —
NO real model, Ollama, or network access needed.

Run:  python -m tests.test_serving
"""
from __future__ import annotations

import asyncio
import time

from app.backends import MockBackend
from app.batcher import DynamicBatcher
from app.cache import EmbeddingCache


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


# --- cache ---------------------------------------------------------------

def test_cache_hit_and_miss() -> None:
    cache = EmbeddingCache(max_size=10, ttl_seconds=60)
    _check("miss on empty cache", cache.get("hello") is None)
    cache.set("hello", [1.0, 2.0])
    _check("hit after set", cache.get("hello") == [1.0, 2.0])
    _check("distinct key still misses", cache.get("world") is None)


def test_cache_ttl_expiry() -> None:
    cache = EmbeddingCache(max_size=10, ttl_seconds=0.05)
    cache.set("hello", [1.0])
    _check("hit before ttl expires", cache.get("hello") == [1.0])
    time.sleep(0.08)
    _check("miss after ttl expires", cache.get("hello") is None)


def test_cache_max_size_zero_disables_caching_without_crashing() -> None:
    # CACHE_MAX_SIZE=0 is how the load test isolates batching from caching —
    # it must degrade to "always miss", not raise.
    cache = EmbeddingCache(max_size=0, ttl_seconds=60)
    cache.set("hello", [1.0])
    _check("set() on a disabled cache is a silent no-op", cache.get("hello") is None)


def test_cache_lru_eviction() -> None:
    cache = EmbeddingCache(max_size=2, ttl_seconds=60)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.set("c", [3.0])  # evicts "a", the least recently used
    _check("oldest entry evicted", cache.get("a") is None)
    _check("newer entry survives", cache.get("c") == [3.0])


# --- mock backend ----------------------------------------------------------

def test_mock_backend_deterministic_and_shaped() -> None:
    backend = MockBackend(dim=16, base_latency_ms=0, per_item_latency_ms=0)
    [v1] = asyncio.run(backend.embed_batch(["hello world"]))
    [v2] = asyncio.run(backend.embed_batch(["hello world"]))
    _check("same text -> identical vector", v1 == v2)
    _check("vector has configured dimension", len(v1) == 16)

    [v3] = asyncio.run(backend.embed_batch(["a completely different sentence"]))
    _check("different text -> different vector", v1 != v3)


def test_mock_backend_batching_is_cheaper_per_item() -> None:
    backend = MockBackend(dim=8, base_latency_ms=10, per_item_latency_ms=10)

    start = time.perf_counter()
    asyncio.run(backend.embed_batch(["x"] * 8))
    batched_s = time.perf_counter() - start

    async def one_at_a_time() -> None:
        for _ in range(8):
            await backend.embed_batch(["x"])

    start = time.perf_counter()
    asyncio.run(one_at_a_time())
    sequential_s = time.perf_counter() - start

    _check(
        "one batched call is faster than 8 sequential calls",
        batched_s < sequential_s,
    )


# --- dynamic batcher ---------------------------------------------------------

class CountingBackend:
    """Wraps a real backend but records how many embed_batch calls were made
    and with what batch sizes, so tests can assert on coalescing behaviour."""

    def __init__(self, inner: MockBackend) -> None:
        self.inner = inner
        self.calls: list[int] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return await self.inner.embed_batch(texts)


def test_batcher_coalesces_concurrent_requests() -> None:
    async def run() -> CountingBackend:
        backend = CountingBackend(MockBackend(dim=8, base_latency_ms=0, per_item_latency_ms=0))
        batcher = DynamicBatcher(backend, max_size=16, max_wait_ms=50)
        batcher.start()
        try:
            # Fire 5 requests "at once" -> they should land in the same batch
            # window and become exactly one backend call of size 5.
            results = await asyncio.gather(*(batcher.submit(f"text-{i}") for i in range(5)))
            _check("all 5 requests got a result", len(results) == 5)
            _check("results have the configured dimension", all(len(r) == 8 for r in results))
        finally:
            await batcher.stop()
        return backend

    backend = asyncio.run(run())
    _check("exactly one backend call was made", backend.calls == [5])


def test_batcher_respects_max_size() -> None:
    async def run() -> CountingBackend:
        backend = CountingBackend(MockBackend(dim=4, base_latency_ms=0, per_item_latency_ms=0))
        batcher = DynamicBatcher(backend, max_size=3, max_wait_ms=50)
        batcher.start()
        try:
            await asyncio.gather(*(batcher.submit(f"text-{i}") for i in range(7)))
        finally:
            await batcher.stop()
        return backend

    backend = asyncio.run(run())
    _check("no batch exceeds max_size", all(size <= 3 for size in backend.calls))
    _check("all 7 requests were served", sum(backend.calls) == 7)


def test_batcher_lone_request_does_not_block_forever() -> None:
    async def run() -> float:
        backend = CountingBackend(MockBackend(dim=4, base_latency_ms=0, per_item_latency_ms=0))
        batcher = DynamicBatcher(backend, max_size=16, max_wait_ms=30)
        batcher.start()
        try:
            start = time.perf_counter()
            await batcher.submit("only request")
            return time.perf_counter() - start
        finally:
            await batcher.stop()

    elapsed_s = asyncio.run(run())
    # Should resolve at ~max_wait_ms once no one else joins the batch, not hang.
    _check("lone request resolves within ~2x max_wait_ms", elapsed_s < 0.06)


def main() -> None:
    print("cache tests:")
    test_cache_hit_and_miss()
    test_cache_ttl_expiry()
    test_cache_max_size_zero_disables_caching_without_crashing()
    test_cache_lru_eviction()

    print("mock backend tests:")
    test_mock_backend_deterministic_and_shaped()
    test_mock_backend_batching_is_cheaper_per_item()

    print("dynamic batcher tests:")
    test_batcher_coalesces_concurrent_requests()
    test_batcher_respects_max_size()
    test_batcher_lone_request_does_not_block_forever()

    print("\nALL SERVING UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
