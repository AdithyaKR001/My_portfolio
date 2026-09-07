"""Async load generator for the /embed endpoint.

Fires CONCURRENCY requests at a time against MODEL_SERVER_URL for
DURATION_SECONDS, drawing texts from a fixed pool of TEXT_POOL_SIZE strings so
that repeats (and therefore cache hits) happen at a realistic rate, and
reports client-observed latency percentiles, throughput, error rate, and the
server-reported cache hit ratio.

Run:  python -m loadtest.run_loadtest
Config is env-driven (see .env.example) so the same script can be pointed at
a batching-enabled server and a batching-disabled one for a before/after
comparison — see `make loadtest-compare`.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8000")
CONCURRENCY = int(os.environ.get("LOADTEST_CONCURRENCY", "20"))
DURATION_SECONDS = float(os.environ.get("LOADTEST_DURATION_SECONDS", "15"))
TEXT_POOL_SIZE = int(os.environ.get("LOADTEST_TEXT_POOL_SIZE", "200"))
LABEL = os.environ.get("LOADTEST_LABEL", "run")

TEXT_POOL = [
    f"incident report {i}: service failed to respond within the configured timeout"
    for i in range(TEXT_POOL_SIZE)
]


@dataclass
class Results:
    latencies_s: list[float] = field(default_factory=list)
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    def summary(self) -> dict:
        total_requests = len(self.latencies_s) + self.errors
        wall_s = max(self.finished_at - self.started_at, 1e-9)
        sorted_lat = sorted(self.latencies_s)

        def pct(p: float) -> float:
            if not sorted_lat:
                return 0.0
            idx = min(int(len(sorted_lat) * p), len(sorted_lat) - 1)
            return sorted_lat[idx]

        return {
            "label": LABEL,
            "concurrency": CONCURRENCY,
            "duration_target_s": DURATION_SECONDS,
            "wall_time_s": round(wall_s, 3),
            "total_requests": total_requests,
            "errors": self.errors,
            "throughput_rps": round(total_requests / wall_s, 2),
            "latency_ms": {
                "mean": round(statistics.mean(self.latencies_s) * 1000, 2) if self.latencies_s else None,
                "p50": round(pct(0.50) * 1000, 2),
                "p95": round(pct(0.95) * 1000, 2),
                "p99": round(pct(0.99) * 1000, 2),
                "max": round(max(sorted_lat) * 1000, 2) if sorted_lat else None,
            },
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": round(
                self.cache_hits / max(self.cache_hits + self.cache_misses, 1), 3
            ),
        }


async def _worker(client: httpx.AsyncClient, deadline: float, results: Results, rng_seed: int) -> None:
    import random

    rng = random.Random(rng_seed)
    while time.perf_counter() < deadline:
        text = rng.choice(TEXT_POOL)
        start = time.perf_counter()
        try:
            resp = await client.post(f"{MODEL_SERVER_URL}/embed", json={"texts": [text]})
            resp.raise_for_status()
            data = resp.json()
            results.latencies_s.append(time.perf_counter() - start)
            results.cache_hits += data["cache_hits"]
            results.cache_misses += data["cache_misses"]
        except Exception:
            results.errors += 1


async def run() -> dict:
    results = Results()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{MODEL_SERVER_URL}/health")
        resp.raise_for_status()

        results.started_at = time.perf_counter()
        deadline = results.started_at + DURATION_SECONDS
        await asyncio.gather(
            *(_worker(client, deadline, results, seed) for seed in range(CONCURRENCY))
        )
        results.finished_at = time.perf_counter()

    return results.summary()


def main() -> None:
    print(f"Load testing {MODEL_SERVER_URL} — concurrency={CONCURRENCY}, "
          f"duration={DURATION_SECONDS}s, label={LABEL!r}")
    summary = asyncio.run(run())
    print(json.dumps(summary, indent=2))

    out_dir = Path(__file__).resolve().parent
    report_path = out_dir / "report.json"
    existing: list[dict] = []
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if not isinstance(existing, list):
            existing = [existing]
    existing.append(summary)
    report_path.write_text(json.dumps(existing, indent=2))
    print(f"\nAppended to {report_path}")


if __name__ == "__main__":
    main()
