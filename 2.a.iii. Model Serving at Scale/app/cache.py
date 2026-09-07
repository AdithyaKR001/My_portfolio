import hashlib

from cachetools import TTLCache

from app.metrics import cache_size


class EmbeddingCache:
    """LRU + TTL cache keyed on a hash of the input text.

    Bounded by max_size (evicts least-recently-used) and ttl_seconds (evicts
    stale entries), so it can't grow unbounded and won't serve embeddings
    computed against a since-swapped model version forever.
    """

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._store: TTLCache = TTLCache(maxsize=max_size, ttl=ttl_seconds)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self._store.get(self._key(text))

    def set(self, text: str, embedding: list[float]) -> None:
        # CACHE_MAX_SIZE=0 is a documented way to disable caching entirely;
        # TTLCache rejects any insert in that case, so treat it as a no-op
        # rather than letting it surface as a 500.
        if self._store.maxsize == 0:
            return
        self._store[self._key(text)] = embedding
        cache_size.set(len(self._store))

    def __len__(self) -> int:
        return len(self._store)
