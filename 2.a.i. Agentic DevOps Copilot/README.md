# Agentic DevOps / Engineering Copilot

Portfolio Project 1 from the AI/ML transition plan. A tool-using LLM agent that
helps engineers track CI/CD pipelines, diagnose build/test failures, inspect
resource usage, and manage tickets — the productionizable evolution of an
enterprise DevOps agent (e.g. an SAP Joule-style assistant).

It runs **fully locally** against **mock DevOps data**, so you need no real CI/CD
or ticketing credentials to try it.

## What it demonstrates (portfolio value)

- **Agentic design**: planning + tool-routing loop (LangGraph ReAct agent), not a
  single prompt.
- **Tool use / function calling** over a realistic DevOps toolset.
- **Human-in-the-loop (HITL)** on write actions (`create_ticket`) — the agent must
  get explicit approval before changing anything.
- **Clean separation** of agent / tools / data, so tools can be swapped from mock
  JSON to real APIs without touching the agent.

## Architecture

```
User ──> Chat UI (Streamlit) / FastAPI ──> LangGraph Agent (LLM + tool loop)
                                              │
                        ┌─────────────────────┼───────────────────────┐
                        ▼           ▼          ▼            ▼
                 list/status   build logs   tickets     resource usage
                        └─────────────────────┼───────────────────────┘
                                              ▼
                             Mock DevOps data (local JSON)
                               (pluggable → real CI/CD, ITSM, metrics APIs)

   create_ticket (write) ── routed through a Human-in-the-loop approval gate
```

## The data (what it uses, and where)

Everything lives in `data/` as plain JSON — read it, edit it, extend it:

| File | Contents |
|---|---|
| `pipelines.json` | 5 CI/CD pipelines with stages, statuses (2 failing, 1 running), commits |
| `logs.json` | build/test logs per pipeline (with real-looking error traces for the failures) |
| `resources.json` | CPU / memory / build-minutes / est. cost per pipeline |
| `tickets.json` | existing engineering tickets (written to when a ticket is created) |

The two failing pipelines are the interesting ones: `pipe-1001` (integration-test
DB timeout) and `pipe-1004` (missing `llm-router` dependency). Ask the agent to
diagnose them.

## Setup

Requires Python 3.10+ and (for the default local mode) [Ollama](https://ollama.com).

**1. Install and start Ollama, pull a tool-capable model:**

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1        # must support tool calling (llama3.1, qwen2.5, mistral-nemo)
ollama serve                # usually already running after install
```

**2. Set up the project:**

```bash
cd agentic-devops-copilot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # defaults to Ollama — no edits needed
```

That's it — the default `.env` is already configured for Ollama, no API key required.

> Prefer OpenAI instead? In `.env` comment out the Ollama block and uncomment
> Option B, then set `OPENAI_API_KEY`. No code change needed.

> Tip: small models can be inconsistent at tool calling. If the agent ignores tools
> or invents ids, try `ollama pull qwen2.5` and set `OLLAMA_MODEL=qwen2.5`.

## Run it

Verify the data/tools with no model or key first:

```bash
python test_tools.py
```

Then chat with the agent:

```bash
python -m app.cli                       # terminal chat (HITL prompts on the CLI)
# or
uvicorn app.api:app --reload            # REST API at http://127.0.0.1:8000/docs
# or
streamlit run streamlit_app.py          # browser chat UI
```

### Run the whole stack with Docker (app + Streamlit + Ollama, one command)

No local Python or Ollama install needed — just Docker:

```bash
docker compose up --build
```

This starts the Ollama server, auto-pulls the model, then starts the API at
http://localhost:8000/docs and the Streamlit chat UI at http://localhost:8501.
Models persist in a named volume, so subsequent starts are fast. Use a different
model with:

```bash
OLLAMA_MODEL=qwen2.5 docker compose up --build
```

First run downloads the model (a few GB) and will take a few minutes.

### Try these

- "Which pipelines are failing right now?"
- "Why did pipe-1001 fail?  What should I do?"
- "Compare the resource cost of pipe-1005 and pipe-1002."
- "Are there any open tickets for pipe-1004?"
- "Open a ticket for the pipe-1004 build failure."  → triggers the HITL approval

## Human-in-the-loop

`create_ticket` is a write action. In the **CLI** you're asked to approve on the
terminal. In the **API**, writes are refused unless you set `AUTO_CONFIRM_WRITES=1`
(demo only). In **Streamlit**, toggle "Auto-approve write actions" in the sidebar.
For production you'd replace this with a durable LangGraph `interrupt` + a proper
approve endpoint.

## Going from mock to real (next step)

Each function in `app/tools.py` is the only thing that touches data. To point at
real systems, replace the `_load(...)` calls with API clients — e.g. GitHub
Actions / GitLab / Jenkins for pipelines and logs, Jira/ServiceNow for tickets,
Prometheus/Grafana for resource usage. The agent, prompt, and UIs stay unchanged.

## Where this leads

- **Project 2 (RAG service)**: add retrieval over runbooks/wikis so the agent
  grounds its diagnoses in your docs.
- **Project 3 (eval & observability)**: wrap this agent with tracing (Langfuse),
  a task-based eval suite, and guardrails — turning the demo into a
  production-grade, measured system.
```
