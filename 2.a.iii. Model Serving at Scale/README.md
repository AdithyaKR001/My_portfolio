# Model Serving at Scale (mini)

Portfolio Project 3 from the AI/ML transition plan. A focused demonstration of
scalable, observable model inference: a FastAPI embedding service with
server-side **request batching**, an **LRU+TTL cache**, **Prometheus metrics**,
a **Grafana dashboard**, and a **Kubernetes HorizontalPodAutoscaler** — plus a
load test that proves the batching/caching actually help, with real numbers.

It runs **fully locally** with a dependency-free mock backend (no model
download, no API key), and is pluggable to a real local Ollama embedding
model with one env var.

## What it demonstrates (portfolio value)

- **Dynamic server-side batching**: concurrent single-item requests are
  coalesced into one backend call, amortizing per-call model overhead.
- **A real before/after load test**, not a claim: `loadtest/report.md` runs
  the *same* workload against naive / batching-only / batching+cache configs
  and reports throughput and p50/p95/p99 latency for each.
- **Production observability**: Prometheus metrics for request rate, cache
  hit ratio, batch size, backend latency, and queue depth, visualized in an
  auto-provisioned Grafana dashboard.
- **Kubernetes autoscaling**: a `Deployment` + `HorizontalPodAutoscaler`
  (`k8s/`) that scales replicas on CPU utilization, with the trade-offs of
  that choice written up in `DESIGN.md`.
- **System-design narrative backed by real code**: `DESIGN.md` explains what
  batching, caching, and autoscaling trade off against each other, and what's
  deliberately left out of a "mini" scope.

## Architecture

```
                 ┌────────────────────── FastAPI app ──────────────────────┐
POST /embed ──►  │  cache lookup (hit? return) ──► DynamicBatcher queue     │  ──► Prometheus /metrics
                 │       │ miss                          │ coalesces         │
                 │       └──────────────► batch of N ◄────┘  up to           │
                 │                             │        BATCH_MAX_SIZE       │
                 │                             ▼        or MAX_WAIT_MS       │
                 │                    EmbeddingBackend.embed_batch()         │
                 │                 (MockBackend, or OllamaBackend)           │
                 └───────────────────────────────────────────────────────────┘
                                             │
                          Prometheus scrapes every replica (dns_sd) ──► Grafana

Kubernetes:  Deployment (N replicas) ◄── HorizontalPodAutoscaler (CPU util)
```

## The load test (before/after)

`python -m loadtest.compare` runs the identical workload (20 concurrent
clients, a 200-string pool with realistic repeats) against three separately
configured server instances, then writes `loadtest/report.md`. Checked-in
result from this machine:

| Label | Throughput (req/s) | p50 (ms) | p95 (ms) | Cache hit ratio |
|---|---:|---:|---:|---:|
| naive (no batching, no cache) | ~66 | ~304 | ~309 | 0.0 |
| batching only | ~127 | ~154 | ~232 | 0.0 |
| batching + cache | ~580 | ~16 | ~119 | 0.98 |

Same mock model, same machine, same traffic — only the serving strategy
changes. See `DESIGN.md` for why each piece contributes what it does, and
`loadtest/report.md` / `loadtest/report.json` for the exact run this table is
drawn from.

## Setup

```bash
cd model-serving-at-scale
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # defaults to the mock backend — nothing else required
```

## Run it

```bash
make test                    # deterministic unit tests (cache, batcher, mock backend)
make run                     # start the API on :8000
```

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["pipeline pipe-1001 failing with DB timeout"]}'
curl -s http://localhost:8000/metrics | head -20
```

```bash
make loadtest                # single run against whatever's on :8000
make loadtest-compare         # the 3-scenario before/after comparison above
```

### One-command Docker stack (app + Prometheus + Grafana)

```bash
docker compose up --build
```

Serves the API at `http://localhost:8000`, Prometheus at
`http://localhost:9090`, and Grafana at `http://localhost:3000` (anonymous
viewer access enabled — no login needed) with the dashboard above
pre-provisioned.

To see multi-replica scraping and load-balancing locally:

```bash
docker compose up -d --scale model-server=3 --no-recreate model-server
docker compose run --rm loadtest   # hits the model-server service name;
                                    # Docker's embedded DNS round-robins it
```

Prometheus's `model-server` job re-resolves DNS every 15s (see
`prometheus.yml`), so all 3 replicas show up as separate scrape targets —
watch "Replicas up" on the Grafana dashboard change as you scale.

### Point it at a real model instead of the mock

```bash
ollama pull nomic-embed-text
ollama serve
```

then in `.env`: `BACKEND=ollama` (defaults already point at
`http://localhost:11434`). Everything else — batching, caching, metrics,
autoscaling — is unchanged; only `app/backends.py::OllamaBackend` is doing
different work underneath.

## Kubernetes (the production autoscaling path)

```bash
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```

Builds on a real cluster with a metrics-server installed for the HPA to read
CPU utilization from. See `DESIGN.md` for why CPU is the shipped default and
what a request-rate-based (KEDA) autoscaler would look like instead.

## Where this fits

Pairs with the Agentic DevOps Copilot and Incident-RAG projects above: this is
the serving layer either of them would sit behind in production — the same
`embed()` call Incident-RAG's ingestion pipeline makes, but now batched,
cached, metered, and horizontally scalable instead of a single unmanaged
model call.
