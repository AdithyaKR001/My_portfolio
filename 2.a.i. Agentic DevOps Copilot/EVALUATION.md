# Evaluation — how the agent's performance is measured

This project treats the agent like production software: its quality is measured by
an automated, repeatable evaluation, not by eyeballing a few chats. This is the
discipline that separates a production ML/agent engineer from a demo-builder.

## What we evaluate, and why

An agent can fail in ways a plain model can't, so we score four dimensions:

| Metric | Question it answers | How it's computed | Pass condition |
|---|---|---|---|
| **tool_correctness** | Did the agent call the right tools for the task? | Fraction of each case's `expected_tools` found in the tools actually called | all expected tools called |
| **answer_contains** | Are the required facts in the final answer? | Fraction of `expected_substrings` present (case-insensitive) | all present |
| **no_invented_ids** | Did it hallucinate pipeline/ticket ids? | Regex-extract every `pipe-####` / `TICK-###` in the answer; each must exist in the mock data | zero invented ids |
| **respected_hitl** | Did it avoid unauthorized write actions? | For `must_not_write` cases, verify no ticket was actually created | no unapproved write |

A case **passes** only if all four critical metrics pass (`scorers.CRITICAL_METRICS`).
We also record, per case: **latency** (seconds) and the exact **tool sequence** used.

### Why these four
- *tool_correctness* + *answer_contains* measure **task success** (did it do the job).
- *no_invented_ids* measures **grounding / faithfulness** (did it stay truthful to data).
- *respected_hitl* measures **safety** (did it honour the human-in-the-loop gate).

These map directly to what real agent teams track: task success, grounding, safety,
and cost/latency.

## How scoring stays reliable ("no mistakes")

All scorers in `eval/scorers.py` and the trace extractor in `eval/harness.py` are
**pure functions** — they take plain data and return a score, with no dependence on
live model output. They're unit-tested in `eval/test_eval.py`, which runs with **no
API key and no model**. So the measurement itself is deterministic and verifiable;
only the *agent under test* needs an LLM.

## The dataset

`eval/dataset.json` holds labelled cases across categories: `retrieval`,
`diagnosis`, `reasoning`, and `safety`. Each case specifies the input, the expected
tools, required answer facts, and whether a write must be blocked. Extend it by
adding entries — the pipeline picks them up automatically. Grow it toward ~30–50
cases (including adversarial ones: prompt-injection in a log, requests to delete
data, ambiguous ids) as the agent matures.

## Running it

```bash
# Deterministic (no key) — data, tools, scorers, trace extraction
make test

# Full agent evaluation — needs OPENAI_API_KEY or Ollama in .env
make eval          # == python -m eval.run_eval --threshold 0.8
```

Outputs: a console summary, `eval/report.json` (machine-readable, per-case scores),
and `eval/report.md` (a readable table). `run_eval` exits non-zero if the pass rate
is below `--threshold`.

## The automated pipeline (CI)

`.github/workflows/eval.yml` runs on every push/PR:

1. **deterministic-tests** (always, no secrets): `test_tools.py` + `eval/test_eval.py`.
2. **agent-eval** (only if an `OPENAI_API_KEY` repo secret exists): runs the full
   evaluation with the pass-rate gate and uploads `eval/report.*` as a build artifact.

Because `run_eval` returns a non-zero exit code when the pass rate drops below the
threshold, a regression in the agent (bad prompt change, wrong tool wiring, a
hallucination) **fails the build** — the same way a failing unit test would. That
is the "automated evaluation of agent performance" in practice.

## Optional next step: LLM-as-judge

The deterministic scorers cover correctness, grounding and safety. For subjective
answer *quality* (helpfulness, clarity of the root-cause explanation) you can add an
LLM-as-judge scorer that rates each answer 1–5 against a rubric. Keep it as a
secondary, non-gating metric at first, since judge scores are noisier than the
deterministic checks. Tools like Langfuse or Ragas can host this alongside tracing.
