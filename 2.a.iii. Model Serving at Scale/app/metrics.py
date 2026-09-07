from prometheus_client import Counter, Gauge, Histogram

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "End-to-end latency of /embed requests, as seen by the client.",
    ["endpoint"],
)

embed_requests_total = Counter(
    "embed_requests_total",
    "Individual text embedding requests, split by cache outcome.",
    ["cache"],
)

embed_batch_size = Histogram(
    "embed_batch_size",
    "Number of texts coalesced into each backend call.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

backend_call_duration_seconds = Histogram(
    "backend_call_duration_seconds",
    "Time spent inside a single backend.embed_batch() call.",
)

embed_queue_depth = Gauge(
    "embed_queue_depth",
    "Requests currently waiting in the batching queue.",
)

cache_size = Gauge(
    "cache_size",
    "Current number of entries in the embedding cache.",
)
