"""Synthetic data generator for the Incident-RAG knowledge base.

Produces a large, varied corpus of DevOps documents plus a *labelled* golden
Q&A set for evaluation. Everything is deterministic (fixed seed), so the gold
relevance labels stay stable and reproducible.

Outputs (JSONL):
  data/corpus/incidents.jsonl   ~ resolved incident write-ups (symptoms/root cause/fix)
  data/corpus/runbooks.jsonl    ~ one runbook per failure category
  data/eval/golden_qa.jsonl     ~ questions -> relevant doc ids + expected answer facts

Run:  python scripts/generate_data.py --incidents 220
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
EVAL = ROOT / "data" / "eval"

SERVICES = [
    "s4-public-cloud-core", "dmc-insights-service", "joule-agent-gateway",
    "hdlfs-archiving-job", "landscape-orchestrator", "sales-cloud-odata",
    "iot-tenant-onboarding", "supply-chain-monitor", "btp-connectivity-proxy",
    "hana-replication-worker",
]

DB_HOSTS = ["hana-test:30015", "hana-stage:30015", "pg-ci:5432", "redis-cache:6379"]
STAGES = ["build", "unit-tests", "integration-tests", "deploy-staging", "smoke-tests"]

# Each category is a template with slot fillers. `question` is authored FROM the
# same content, which guarantees the golden label (the incident id) is correct.
CATEGORIES = [
    {
        "key": "integration_test_timeout",
        "title": "Integration test timeout against {db}",
        "symptoms": "{test} fails with SocketTimeoutException (read timed out after 30000ms) while connecting to {db} during the integration-tests stage.",
        "root_cause": "The test DB connection pool is exhausted under parallel test load, so new connections block past the 30s timeout.",
        "resolution": "Increase the HikariCP pool size to 20, or run the integration suite with --threads 1 to serialize DB access; add a connection-acquire retry with backoff.",
        "tags": ["timeout", "database", "integration-tests", "connection-pool"],
        "slots": {"test": ["OrderServiceIT.shouldReplicateOrder", "InventoryIT.shouldSync", "BillingIT.shouldPost"], "db": DB_HOSTS},
        "question": "Our {test} keeps timing out against {db} in CI. How do we fix it?",
        "answer_keywords": ["pool", "timeout", "serialize"],
    },
    {
        "key": "dependency_not_found",
        "title": "Build fails: cannot resolve {artifact}:{version}",
        "symptoms": "The build stage fails with 'Could not resolve dependency {artifact}:{version}; latest in artifactory is {prev}'.",
        "root_cause": "The pom/build file pins {artifact} to {version}, a version that was never published to artifactory.",
        "resolution": "Pin {artifact} back to {prev}, or publish {version} to artifactory; then re-run the build.",
        "tags": ["build", "dependency", "artifactory", "version"],
        "slots": {"artifact": ["com.sap.ai:llm-router", "com.sap.dmc:order-model", "com.sap.hana:jdbc"],
                  "version": ["2.4.1", "3.1.0", "5.2.7"], "prev": ["2.3.9", "3.0.8", "5.2.6"]},
        "question": "The build can't resolve {artifact} version {version}. What's the fix?",
        "answer_keywords": ["pin", "publish", "artifactory"],
    },
    {
        "key": "oom_kill",
        "title": "Pod OOMKilled during {stage}",
        "symptoms": "The {service} pod is OOMKilled during the {stage} stage; kubelet reports memory limit exceeded (peak > {mem}Gi).",
        "root_cause": "The container memory limit is set below the working-set needed to process the current dataset size.",
        "resolution": "Raise the container memory limit to {newmem}Gi and set -Xmx to 75% of it; enable heap-dump-on-OOM to confirm.",
        "tags": ["oom", "memory", "kubernetes", "limits"],
        "slots": {"stage": STAGES, "mem": ["4", "8", "12"], "newmem": ["8", "16", "24"]},
        "question": "The {service} pod gets OOMKilled in {stage}. How do we resolve it?",
        "answer_keywords": ["memory limit", "xmx", "raise"],
    },
    {
        "key": "flaky_test",
        "title": "Flaky test {test} fails intermittently",
        "symptoms": "{test} passes locally but fails ~15% of CI runs with an assertion on unexpected ordering.",
        "root_cause": "A race condition: the test asserts on results before an async write has completed.",
        "resolution": "Await the async completion (or poll with a deadline) before asserting; remove the fixed sleep and use awaitility.",
        "tags": ["flaky", "race-condition", "async", "tests"],
        "slots": {"test": ["EventOrderingTest", "CacheEvictionTest", "WebhookDeliveryTest"]},
        "question": "{test} is flaky and fails intermittently in CI. What causes it and how do we fix it?",
        "answer_keywords": ["race", "await", "async"],
    },
    {
        "key": "cert_expiry",
        "title": "TLS handshake fails for {service}",
        "symptoms": "Outbound calls from {service} fail with 'certificate expired' / PKIX path validation errors starting {date}.",
        "root_cause": "The client mTLS certificate expired and was not rotated ahead of its expiry.",
        "resolution": "Rotate the mTLS certificate, redeploy the secret, and add an alert 30 days before cert expiry to prevent recurrence.",
        "tags": ["tls", "certificate", "mtls", "expiry"],
        "slots": {"date": ["2026-06-01", "2026-07-15", "2026-08-02"]},
        "question": "{service} started failing TLS handshakes with expired certificate errors. How do we fix and prevent this?",
        "answer_keywords": ["rotate", "certificate", "alert"],
    },
    {
        "key": "disk_full",
        "title": "Build fails with 'no space left on device'",
        "symptoms": "The {stage} stage fails with 'No space left on device' on the CI runner.",
        "root_cause": "The build artifact and Docker layer cache filled the runner's disk; nothing prunes it.",
        "resolution": "Add a cache-pruning step (docker system prune, clear ~/.m2 older than 7d) and increase the runner disk to {disk}GB.",
        "tags": ["disk", "storage", "cache", "runner"],
        "slots": {"stage": STAGES, "disk": ["100", "200", "500"]},
        "question": "CI fails with 'no space left on device' in the {stage} stage. How do we fix it?",
        "answer_keywords": ["prune", "cache", "disk"],
    },
    {
        "key": "deploy_readiness",
        "title": "deploy-staging times out on readiness probe",
        "symptoms": "The deploy-staging stage for {service} times out; the pod never becomes Ready and rolls back.",
        "root_cause": "The readiness probe's initialDelaySeconds is shorter than the service's real startup time, so it's marked unhealthy too early.",
        "resolution": "Increase the readiness probe initialDelaySeconds to {delay}s and add a startupProbe so slow starts aren't killed.",
        "tags": ["deploy", "readiness-probe", "kubernetes", "startup"],
        "slots": {"delay": ["30", "60", "90"]},
        "question": "Deployment of {service} to staging keeps timing out on the readiness probe. What's the fix?",
        "answer_keywords": ["readiness", "initialdelay", "startupprobe"],
    },
    {
        "key": "kafka_lag",
        "title": "Consumer lag spikes on {topic}",
        "symptoms": "Consumer lag on topic {topic} spikes into the millions; {service} falls behind and alerts fire.",
        "root_cause": "Partition key skew concentrates traffic on a few partitions, so a subset of consumers is saturated.",
        "resolution": "Rebalance by increasing partitions to {parts} and choosing a higher-cardinality partition key; scale the consumer group.",
        "tags": ["kafka", "consumer-lag", "partitions", "throughput"],
        "slots": {"topic": ["order-events", "sensor-readings", "audit-log"], "parts": ["12", "24", "48"]},
        "question": "Kafka consumer lag on {topic} is spiking. What causes it and how do we fix it?",
        "answer_keywords": ["partition", "rebalance", "skew"],
    },
    {
        "key": "config_missing",
        "title": "{service} crashloops: missing config {key}",
        "symptoms": "{service} crash-loops on startup with 'required property {key} is not set'.",
        "root_cause": "A new required env var {key} was added in code but not added to the deployment's config map.",
        "resolution": "Add {key} to the config map / secret for every environment and redeploy; add a startup config-validation check.",
        "tags": ["config", "crashloop", "environment", "startup"],
        "slots": {"key": ["LLM_ROUTER_URL", "HANA_DSN", "KAFKA_BROKERS"]},
        "question": "{service} is crashlooping because config {key} is missing. How do we fix it?",
        "answer_keywords": ["config map", "add", "redeploy"],
    },
    {
        "key": "rate_limit",
        "title": "429s from {dep} throttle {service}",
        "symptoms": "{service} sees a surge of HTTP 429 responses from {dep}, causing cascading request failures.",
        "root_cause": "No client-side rate limiting or backoff, so retries amplify load past the {dep} quota.",
        "resolution": "Add token-bucket client rate limiting and exponential backoff with jitter; request a higher {dep} quota if needed.",
        "tags": ["rate-limit", "429", "backoff", "resilience"],
        "slots": {"dep": ["openai-api", "s3", "artifactory", "hana-cloud"]},
        "question": "{service} is getting throttled with 429s from {dep}. How do we handle it?",
        "answer_keywords": ["backoff", "rate limit", "jitter"],
    },
]


def _fill(text: str, slot_values: dict, service: str) -> str:
    return text.format(service=service, **slot_values)


def generate(n_incidents: int, seed: int = 7):
    rng = random.Random(seed)
    incidents, runbooks, golden = [], [], []

    # One runbook per category (a stable reference doc for that failure class).
    for ci, cat in enumerate(CATEGORIES):
        rb_id = f"RB-{100 + ci}"
        runbooks.append({
            "id": rb_id,
            "type": "runbook",
            "category": cat["key"],
            "title": f"Runbook: diagnosing and fixing {cat['key'].replace('_', ' ')}",
            "tags": cat["tags"],
            "body": (
                f"# Runbook: {cat['key'].replace('_', ' ')}\n"
                f"Typical symptoms: {cat['symptoms'].replace('{', '').replace('}', '')}\n"
                f"Common root cause: {cat['root_cause']}\n"
                f"Standard resolution: {cat['resolution'].replace('{', '').replace('}', '')}\n"
                f"Escalate to the owning team if the standard resolution does not clear it within 30 minutes."
            ),
        })

    used_signatures = set()
    for i in range(n_incidents):
        cat = CATEGORIES[i % len(CATEGORIES)]
        service = rng.choice(SERVICES)
        slot_values = {k: rng.choice(v) for k, v in cat["slots"].items()}
        signature = (cat["key"], service, tuple(sorted(slot_values.items())))
        if signature in used_signatures:
            continue
        used_signatures.add(signature)

        inc_id = f"INC-{1000 + i}"
        title = _fill(cat["title"], slot_values, service)
        symptoms = _fill(cat["symptoms"], slot_values, service)
        root_cause = _fill(cat["root_cause"], slot_values, service)
        resolution = _fill(cat["resolution"], slot_values, service)
        body = (
            f"# {inc_id}: {title}\n"
            f"Service: {service}\n"
            f"Category: {cat['key']}\n"
            f"Symptoms: {symptoms}\n"
            f"Root cause: {root_cause}\n"
            f"Resolution: {resolution}\n"
        )
        incidents.append({
            "id": inc_id, "type": "incident", "category": cat["key"], "service": service,
            "title": title, "tags": cat["tags"], "symptoms": symptoms,
            "root_cause": root_cause, "resolution": resolution, "body": body,
        })

        # Turn a subset into labelled eval questions. Gold docs = this incident +
        # its category runbook (both are legitimately relevant).
        if i % 4 == 0:
            rb_id = f"RB-{100 + (i % len(CATEGORIES))}"
            golden.append({
                "id": f"Q-{i}",
                "question": _fill(cat["question"], slot_values, service),
                "relevant_doc_ids": [inc_id, rb_id],
                "answer_keywords": cat["answer_keywords"],
                "category": cat["key"],
            })

    return incidents, runbooks, golden


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", type=int, default=220)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    incidents, runbooks, golden = generate(args.incidents, args.seed)
    _write_jsonl(CORPUS / "incidents.jsonl", incidents)
    _write_jsonl(CORPUS / "runbooks.jsonl", runbooks)
    _write_jsonl(EVAL / "golden_qa.jsonl", golden)
    print(f"Wrote {len(incidents)} incidents, {len(runbooks)} runbooks, {len(golden)} golden Q&A.")


if __name__ == "__main__":
    main()
