# Agent Evaluation & Observability Framework

A framework for **testing, tracing, and scoring AI agents** — the "unit
tests + APM dashboard" that agentic systems need but rarely have. Point it
at any agent, run a suite of labeled tasks against it, and get automated
pass/fail scoring plus a visual dashboard of cost, latency, and quality
trends over time — so you catch regressions when you change a prompt or
swap a model, instead of finding out from users.

Built to demonstrate production LLMOps thinking, not just "an agent that
does X": most agent portfolios stop at the demo. This answers the next
question every team building agents actually asks — **"how do we know it
still works after we change something?"**

## Why this exists

Traditional software has unit tests: same input → same output, always.
Agents don't work that way — they're non-deterministic, multi-step, and
call tools. A prompt tweak or model swap can silently break tool routing,
introduce hallucinations, or blow up cost/latency, and you won't notice
until a user does. This framework turns "does my agent still work" into
something you can check automatically, the same way `pytest` checks
regular code.

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌───────────────┐     ┌────────────┐
│ Golden Tasks │ --> │  Agent   │ --> │    Tracer     │ --> │  Database  │
│ (labeled set)│     │ (traced) │     │ (steps, cost, │     │ (SQLite /  │
└─────────────┘     └──────────┘     │  latency)      │     │  Postgres) │
                                       └───────────────┘     └─────┬──────┘
                                                                    │
                     ┌──────────────┐                              │
                     │  Evaluation  │  <---------------------------┘
                     │   Metrics    │
                     │ - tool accuracy
                     │ - keyword groundedness
                     │ - latency budget
                     │ - LLM-judge groundedness
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  Streamlit   │
                     │  Dashboard   │
                     └──────────────┘
```

**Key design choice:** the tracing/storage/evaluation layers don't know
anything about *how* the agent works — only that it produces a `RunTrace`
(query, tool calls, final answer, tokens, timing). That means you can
point this framework at a completely different agent (your RAG assistant,
your parent-communication platform, a CrewAI system, whatever) by writing
one new function, without touching anything else.

## What's included

- **Demo agent** (`src/agent/`) — a small LangGraph ReAct agent with 3
  tools (calculator, internal knowledge-base search, current date), so
  the framework has something real to evaluate out of the box.
- **Golden dataset** (`src/dataset/golden_tasks.json`) — 15 labeled
  tasks with expected tool + expected answer keywords, used for
  regression testing.
- **Tracer** (`src/tracing/`) — framework-agnostic recorder of every
  step an agent takes (tool calls, latency, token usage).
- **Storage** (`src/storage/`) — SQLAlchemy models for runs, steps, and
  eval results. Defaults to SQLite; swap `DATABASE_URL` for a Postgres
  connection string and nothing else changes.
- **Evaluation metrics** (`src/evaluation/`):
  - `tool_accuracy` — did the agent call the tool the task expects?
  - `keyword_groundedness` — cheap check that the final answer contains
    the expected facts (no LLM call needed)
  - `latency_budget` — did the run finish inside an acceptable time?
  - `llm_judge.groundedness` — a second Claude call that checks whether
    the final answer is actually supported by the tool output, catching
    hallucinations keyword matching would miss
- **Runner** (`src/runner.py`) — CLI that runs the whole suite and
  prints a pass/fail table.
- **Dashboard** (`dashboard/app.py`) — Streamlit app: pass-rate trend
  over time, cost/latency scatter, per-metric breakdown, and a full
  step-by-step drill-down into any individual run.
- **Mock mode** — the entire pipeline (agent, judge, everything) can run
  with zero API calls and zero cost via deterministic rule-based
  execution. Use this to test your setup and for a genuinely free CI
  check; switch to `--live` when you want real Claude-API-graded runs.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: add your ANTHROPIC_API_KEY if you want to run --live
```

## Running it

```bash
# 1. Run the eval suite (mock mode — free, no API key needed)
python -m src.runner

# 2. Run it again with a deliberately injected bug, to see regression detection
python -m src.runner --inject-bug

# 3. Run for real against the Claude API (needs ANTHROPIC_API_KEY in .env)
python -m src.runner --live

# 4. Explore results visually
streamlit run dashboard/app.py

# 5. Run the framework's own unit tests
pytest tests/ -v
```

### The regression-detection demo (best interview story)

```bash
python -m src.runner              # ~98% pass rate — baseline
python -m src.runner --inject-bug # ~82% pass rate — simulated bug
streamlit run dashboard/app.py    # see the drop plotted on the trend chart
```

The bug deliberately misroutes calculator questions to the knowledge-base
search tool — simulating what happens in production when a prompt edit or
model swap subtly breaks tool selection. The framework catches it
immediately across `tool_accuracy`, `keyword_groundedness`, and the
LLM-judge check, and the dashboard's pass-rate-over-time chart shows
exactly when and how much quality dropped.

### A real finding this framework already surfaced

Task `T009` ("minimum password length") fails `keyword_groundedness` even
in the *non-buggy* run — the naive word-overlap search tool mis-retrieves
the "Shipping Timelines" doc instead of "Account Security" because generic
words (`is`, `the`, `for`, `account`) create false overlap. This is a real
retrieval-quality bug the framework caught on its own, and it's exactly
the kind of thing that's invisible until you have automated evals. (See
"What to add next" below for the fix.)

## What still needs to be added

This is a solid, working foundation, but here's what would take it from
"portfolio project" to genuinely production-grade — good next steps and
honest talking points for interviews:

1. **Better retrieval in the search tool.** Currently naive word-overlap
   matching (see the T009 finding above). Swap for TF-IDF or embedding
   similarity (you already know ChromaDB from your other projects — reuse
   it here) for meaningfully better groundedness scores.

2. **CI integration.** Add a GitHub Actions workflow that runs
   `python -m src.runner` (mock mode) on every push and fails the build
   if pass rate drops below a threshold — turns this into real automated
   regression testing, not just a manual script.

3. **Statistical significance / flakiness handling.** Agents are
   non-deterministic — a single failed run might be noise. Add "run each
   task N times, require M/N passes" logic before treating something as a
   real regression.

4. **Postgres deployment + Docker.** Swap `DATABASE_URL` to a real
   Postgres instance (e.g. on Render/Railway/Supabase), containerize with
   a `Dockerfile` + `docker-compose.yml` (app + Postgres), and deploy the
   Streamlit dashboard (Streamlit Community Cloud or a small VM). This is
   the single highest-leverage addition for "production-shaped" resume
   credibility.

5. **Point it at a real agent of yours.** Swap the demo agent for your
   [[deep-research-assistant]] or [[parent-communication-platform]] — the
   tracing/eval/dashboard layers already don't care what agent produced
   the trace. This turns the story from "I built an eval framework" into
   "I used my eval framework to catch a real regression in my production
   agent," which is a much stronger interview answer.

6. **More judge dimensions.** Right now the LLM judge only checks
   groundedness. Add judges for tone/format adherence, safety/PII leakage
   checks, and instruction-following, each as its own scored dimension.

7. **Alerting.** A Slack/email webhook that fires when a scheduled eval
   run's pass rate drops below the quality gate — closes the loop from
   "dashboard you have to check" to "system that tells you."

8. **Cost/latency budgets per task type**, not just globally — some
   tasks are legitimately slower/pricier (e.g. multi-tool chains) and a
   single global budget will either be too loose or too strict.

## Tech stack

Python · LangGraph · LangChain · Anthropic Claude API · SQLAlchemy ·
SQLite (Postgres-ready) · Streamlit · Plotly · pytest
