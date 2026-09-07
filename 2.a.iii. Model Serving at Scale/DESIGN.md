# Design — architecture and trade-offs

Scope: this is the "mini" version of a model-serving platform — enough of a real
system to make honest architectural trade-offs (and measure their effect),
without pretending to be a full ML platform. What's deliberately out of scope
is listed at the bottom.

## What's actually being served

A text-embedding endpoint (`POST /embed`) sitting in front of a swappable
backend: a dependency-free `MockBackend` by default (deterministic,
feature-hashed vectors with simulated inference latency, so the whole system
runs and is load-testable with zero setup), or a real `OllamaBackend`
(`nomic-embed-text`, matching the embedding model used by the Incident-RAG
project) for one env var flip. The serving concerns this project is actually
about — batching, caching, autoscaling, observability — are orthogonal to
which backend answers the call, which is the point: they'd look the same in
front of a bigger model.

## Request batching

**What:** `DynamicBatcher` (`app/batcher.py`) holds an incoming request for up
to `BATCH_MAX_WAIT_MS` hoping concurrent requests arrive to batch with it, but
never longer, and never past `BATCH_MAX_SIZE` items — then issues exactly one
`backend.embed_batch()` call for the whole group.

**Why it helps:** most model inference has fixed per-call overhead (device
transfer, framework dispatch) on top of a cost that scales sub-linearly with
batch size. Coalescing N concurrent single-item requests into one batched call
amortizes that overhead across all N. The load test (`loadtest/report.md`)
shows this directly: batching alone (cache disabled) very roughly doubles
throughput at fixed concurrency, using the *exact same* mock model.

**Trade-off:** a lone request now waits up to `BATCH_MAX_WAIT_MS` even when
nothing else is in flight, trading a small amount of tail latency for
throughput under load. `max_wait_ms=20` is a deliberately small number — for a
user-facing endpoint you want the batching window well under human-perceptible
latency; a batch/offline pipeline could trade a much larger window for a much
bigger throughput win. This is a tunable, not a fixed constant, for exactly
that reason.

**What would break it:** wildly different per-request costs (e.g. some texts
10x longer than others) — batching assumes items in a batch cost roughly the
same, and a slow item in a batch delays the fast ones behind it. A production
version would bucket by length before batching.

## Caching

**What:** `EmbeddingCache` (`app/cache.py`) is an LRU + TTL cache
(`cachetools.TTLCache`) keyed on a hash of the input text, checked *before* a
request ever reaches the batcher. `CACHE_MAX_SIZE=0` disables it entirely
(used to isolate batching's effect in the load test) without needing a
separate code path.

**Why it helps:** real incident/support workloads (this backend's actual
intended caller, per the Incident-RAG project) re-ask near-identical questions
constantly. A cache hit skips the model entirely — no queueing, no inference —
so it's strictly cheaper than even the best-batched miss. The load test shows
this as the dominant effect: adding caching on top of batching is what takes
throughput from ~2x baseline to ~9x baseline, because most of the pool's
requests become hits once it's warm.

**Trade-offs:**
- **Bounded, not free.** `CACHE_MAX_SIZE` and `CACHE_TTL_SECONDS` cap memory
  and staleness — an unbounded cache is a memory leak; an unbounded TTL means
  responses silently outlive a model upgrade.
- **In-process, not shared.** Each replica has its own cache, so scaling out
  multiplies memory use and starts every new replica cold, and the same query
  can be a hit on one replica and a miss on another. A production version
  behind real horizontal scale would put this in Redis instead — noted, not
  built, to keep this "mini."
- **Exact-match only.** Keyed on an exact text hash, so near-duplicate phrasing
  doesn't hit. Semantic caching (embed-then-cosine-lookup) buys more hits at
  the cost of needing an embedding just to check the cache — a real
  optimization for a v2, deliberately left out here.

## Autoscaling

**What's shipped:** `k8s/hpa.yaml` — a standard `autoscaling/v2`
`HorizontalPodAutoscaler` scaling `model-server` on CPU utilization (60%
target, 2–10 replicas), which is why `k8s/deployment.yaml` sets CPU
requests/limits (HPA percentages are meaningless without a request to be a
percentage *of*).

**Why CPU, and why that's not the final answer:** CPU utilization needs zero
extra infrastructure — every cluster supports it out of the box — which makes
it the right default to ship. But this workload's real bottleneck signal is
*queue depth / requests-per-second*, not CPU: a replica sitting in
`asyncio.sleep()` waiting on a slow model call looks idle to the CPU metric
while it's actually saturated on concurrency. The honest fix is
request-rate-based autoscaling (e.g. KEDA with a Prometheus scaler reading
`embed_queue_depth` or request rate directly) — flagged here as the
correct next step rather than built, because it needs a metrics adapter and
KEDA installed in-cluster, which is infrastructure this "mini" project doesn't
own.

**Local approximation:** there's no real Kubernetes cluster in this repo's
dev loop, so horizontal scaling is demonstrated locally via
`docker compose up --scale model-server=N` — Docker's embedded DNS
round-robins new connections across replicas for anything that calls the
`model-server` service name (see `loadtest` service in `docker-compose.yml`),
and Prometheus's `dns_sd_configs` (see `prometheus.yml`) re-resolves that same
name so every replica shows up as its own scrape target. That's a genuine,
correct way to exercise multi-replica behavior locally; it's not a
substitute for the real HPA, which needs an actual cluster to prove out.

## Observability

Everything a scaling decision needs is a Prometheus metric, not a log line:
`embed_requests_total{cache}` (traffic, cache effectiveness),
`embed_batch_size` (is batching actually coalescing anything),
`backend_call_duration_seconds` (is the model the bottleneck),
`embed_queue_depth` (backpressure, and the input KEDA would want),
`http_request_duration_seconds` (what the client actually feels). The Grafana
dashboard (`grafana/dashboards/model-serving.json`, auto-provisioned) plots
all of them together against replica count, so a throughput change and its
cause (more replicas? warmer cache? bigger batches?) show up on one screen.

## What's explicitly out of scope ("mini")

- A shared (Redis) cache across replicas.
- Length-aware / priority batching queues.
- A real request-rate-based (KEDA) autoscaler, vs. the CPU-based one shipped.
- TLS, auth, and rate limiting on the API — this is a portfolio demo served
  locally / inside a private cluster network, not a public endpoint.
- GPU scheduling — the mock backend is CPU-bound by design; a real model
  would need `nvidia.com/gpu` resource requests and a GPU-aware autoscaler.
