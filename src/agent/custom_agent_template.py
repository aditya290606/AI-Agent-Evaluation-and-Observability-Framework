"""
Template for connecting YOUR own custom agent to this evaluation framework.

Any agent (CrewAI, AutoGen, LangGraph, custom REST API, etc.) can be evaluated
by this framework as long as it returns a `RunTrace` object.

Steps to evaluate your own agent:
  1. Implement `run_custom_agent(task_id, query)` below.
  2. Add your golden evaluation questions to `src/dataset/golden_tasks.json`.
  3. In `src/runner.py`, import `run_custom_agent` instead of `demo_agent.run_agent`.
  4. Run `python -m src.runner` and inspect results on `streamlit run dashboard/app.py`.
"""

import time
from src.tracing.tracer import RunTrace, TracedStep, StepTimer


def run_custom_agent(task_id: str, query: str) -> RunTrace:
    """Wrap your custom agent call here and return a RunTrace."""
    trace = RunTrace(task_id=task_id, query=query, is_mock=False)

    # --- 1. Call your agent ---
    # Example:
    # response = my_agent.run(query)

    # --- 2. Log any tool calls made by your agent (optional but recommended) ---
    # with StepTimer() as t:
    #     tool_output = my_agent_tool.execute(...)
    # trace.log_step(TracedStep(
    #     step_type="tool_call",
    #     tool_name="your_tool_name",
    #     input_data="input sent to tool",
    #     output_data="output returned by tool",
    #     latency_ms=t.elapsed_ms
    # ))

    # --- 3. Final Answer ---
    final_answer = "Replace with your agent's response string"
    trace.log_step(TracedStep(step_type="final_answer", output_data=final_answer))
    trace.finish(final_answer)

    return trace
