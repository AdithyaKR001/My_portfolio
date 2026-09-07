import asyncio

from app.backends import EmbeddingBackend
from app.metrics import backend_call_duration_seconds, embed_batch_size, embed_queue_depth


class DynamicBatcher:
    """Coalesces concurrent single-text embed requests into batched backend calls.

    A request is held for at most max_wait_ms hoping more requests arrive to
    batch with it, but never longer than that and never past max_size items —
    so batching improves throughput under load without adding unbounded
    latency to a lone request.
    """

    def __init__(self, backend: EmbeddingBackend, max_size: int, max_wait_ms: float) -> None:
        self.backend = backend
        self.max_size = max_size
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((text, fut))
        embed_queue_depth.set(self._queue.qsize())
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self.max_wait_s
            while len(batch) < self.max_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            embed_queue_depth.set(self._queue.qsize())
            embed_batch_size.observe(len(batch))
            texts = [text for text, _ in batch]

            start = loop.time()
            try:
                embeddings = await self.backend.embed_batch(texts)
                backend_call_duration_seconds.observe(loop.time() - start)
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            for (_, fut), embedding in zip(batch, embeddings):
                if not fut.done():
                    fut.set_result(embedding)
