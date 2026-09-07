# RAG Evaluation — what is measured and how

RAG systems fail in two distinct places, so we measure both separately:
**retrieval** (did we fetch the right documents?) and **generation** (did the model
answer correctly and faithfully from them?). Measuring them apart tells you *where*
a regression is — a bad answer from good retrieval is a prompt/model problem; a bad
answer from bad retrieval is an embedding/chunking problem.

## The golden dataset

`data/eval/golden_qa.jsonl` — each row is a question authored **from** a specific
incident, so its `relevant_doc_ids` label (the incident + its category runbook) is
correct by construction. Each row also carries `answer_keywords`: the key facts a
correct answer must contain.

## Retrieval metrics (need only embeddings — run with `--retrieval-only`)

| Metric | Definition | Why it matters |
|---|---|---|
| **hit_rate@k** | 1 if any relevant doc is in the top-k, else 0 (averaged) | the headline "did we find it at all" number; the CI gate metric |
| **recall@k** | fraction of relevant docs found in the top-k | did we get *all* the needed context |
| **precision@k** | fraction of the top-k that are relevant | how much noise is in the context window |
| **MRR** | mean of 1/rank of the first relevant doc | how highly the right doc ranks (reranker quality) |

## Generation metrics (need the LLM — full `run_eval`)

| Metric | Definition | Why it matters |
|---|---|---|
| **answer_match** | fraction of `answer_keywords` present in the answer | did it actually state the correct fix (correctness proxy) |
| **citation_validity** | fraction of doc ids the answer cites that were actually retrieved | **faithfulness proxy** — catches fabricated citations / ungrounded claims |

`citation_validity` works because the prompt forces the model to cite the ids it
used and to say "I don't have that" when the context lacks the answer. A cited id
that wasn't retrieved is a hallucination.

## Why the measurement is trustworthy ("no mistakes")

Every metric in `eval/metrics.py` is a **pure function** and is unit-tested in
`eval/test_eval.py` with hand-checked inputs — **no model or API key required**.
`test_eval.py` also validates the generated golden set is well-formed. So the
scoring itself is deterministic and verifiable; only the system *under test* uses a
model.

## Running

```bash
make test              # deterministic metric unit tests (no model)
make eval-retrieval    # retrieval metrics only (embeddings, fast)
make eval              # full retrieval + generation eval -> eval/report.json
make dashboard         # visualize eval/report.json
```

`run_eval` writes `eval/report.json` (machine-readable, per-case) and
`eval/report.md`, prints a summary, and returns a non-zero exit code when
`hit_rate@k` is below `--threshold`. That exit code is the CI gate.

## The tuning loop (what to actually do with this)

1. Run `make eval-retrieval`, read hit_rate@k / MRR on the dashboard.
2. Change one knob in `.env` — `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANK`,
   or the embedding model — and re-ingest / re-eval.
3. Keep changes that raise recall without tanking precision; then run the full
   `make eval` to confirm generation quality followed.

## Extending

- **LLM-as-judge faithfulness/helpfulness** as a secondary (non-gating) metric.
- **Adversarial cases** in the golden set: unanswerable questions (the model must
  refuse), near-duplicate incidents (tests ranking), and prompt-injection inside a
  retrieved doc (tests robustness).
- Swap the lexical reranker for a cross-encoder and compare MRR.
```
