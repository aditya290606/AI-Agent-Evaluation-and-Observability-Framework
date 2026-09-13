# 🔌 How to Add Your Own Agent & Tasks (AgentPulse SDK)

AgentPulse is **agent-agnostic**: the evaluation metrics, observability flight recorder, and Streamlit dashboard work seamlessly with **any framework** (LangGraph, CrewAI, AutoGen, LlamaIndex, or custom Python/REST agents).

You can integrate your agent in **one of two ways**:
1. **Method A (Recommended): The `agentpulse` SDK** — Instrument your functions with decorators and evaluate in Python.
2. **Method B: CLI Evaluation** — Point `python -m src.runner --callable` directly to your agent function without editing framework code.

---

## Method A: AgentPulse SDK (Fastest & Cleanest)

### Step 1: Instrument Your Tools & Agent Function

Use the lightweight `@agentpulse.trace`, `@agentpulse.tool`, and `@agentpulse.mcp_tool` decorators:

```python
import agentpulse

# 1. Instrument tools
@agentpulse.tool(name="calculator")
def my_calculator(expression: str) -> str:
    return str(eval(expression))

# 2. Instrument MCP tools
@agentpulse.mcp_tool(server="Enterprise Knowledge Base", name="search_documents")
def search_kb(query: str) -> str:
    return "Refunds are processed within 14 days."

# 3. Instrument your agent workflow
@agentpulse.trace(agent_name="MyProductionAgent", version="v1.0", model="claude-3-5-haiku")
def my_agent(query: str) -> str:
    if "refund" in query:
        info = search_kb(query)
        return f"Policy answer: {info}"
    return "I can answer your questions."
```

### Step 2: Run Standalone Live Executions

Any call to an `@agentpulse.trace` decorated function is automatically recorded, timed, cost-accounted, and stored in the database:

```python
# Standalone execution — immediately viewable in the dashboard!
response = my_agent("What is the refund policy?")
print(response)
```

### Step 3: Evaluate Against Golden Benchmarks

Evaluate your agent against the golden benchmark dataset with a single line of Python:

```python
# Run automated benchmark evaluation
report = agentpulse.evaluate(
    agent=my_agent,
    dataset="golden_tasks",
    agent_name="MyProductionAgent",
    agent_version="v1.0",
)

print(f"Overall Quality Score: {report.weighted_score:.1f}%")
print(f"Passed Checks: {report.passed_checks}/{report.total_checks}")
```

---

## Method B: CLI Suite Runner (`--callable`)

You can evaluate any external Python function directly from the terminal without editing any files:

```bash
# Point to your python function (format: module_path:function_name)
python -m src.runner --callable examples.quickstart_external_agent:support_agent

# With custom experiment name and tag filtering
python -m src.runner --callable my_module:my_agent --experiment "v1_release_eval" --tag "rag"
```

---

## Method C: Custom Test Cases & Datasets

To evaluate custom questions:
1. Add tasks to `src/dataset/golden_tasks.json`, or
2. Create your own JSON file (e.g. `my_dataset.json`):

```json
[
  {
    "task_id": "TASK_001",
    "query": "What is the return policy for damaged electronics?",
    "expected_tool": "search_knowledge_base",
    "expected_keywords": ["14 days", "replacement", "refund"]
  }
]
```

Then run:
```bash
python -m src.runner --callable my_module:my_agent --dataset my_dataset.json
```

---

## Launch the Observability Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` to explore:
- **Trace Waterfalls & Hierarchical Trees**: View agent reasoning, child tool spans, and MCP server operations.
- **Regression Detection**: Compare `v1.0` vs `v1.1` versions side-by-side.
- **7-Dimension Tool & MCP Analytics**: Tool selection accuracy, sequence correctness, unnecessary calls.
- **Financials**: Cost per run, token accounting, and P95 latency distributions.
- **AI Copilot**: Ask questions grounded directly in your agent's evaluation telemetry!
