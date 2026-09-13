"""
Basic unit tests for the deterministic metrics. Run with: pytest tests/
This is the "prove the eval framework itself is correct" layer — a
framework whose own metrics aren't tested isn't trustworthy for judging
other agents.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracing.tracer import RunTrace, TracedStep
from src.evaluation import metrics


def _make_trace(tools_called, final_answer, latency_ms=100.0):
    trace = RunTrace(task_id="TEST", query="test query")
    for tool_name in tools_called:
        trace.log_step(TracedStep(step_type="tool_call", tool_name=tool_name, output_data="dummy"))
    trace.finish(final_answer)
    trace.end_time = trace.start_time + (latency_ms / 1000)
    return trace


def test_tool_accuracy_pass():
    trace = _make_trace(["calculator"], "the answer is 42")
    score, passed, _ = metrics.tool_accuracy(trace, "calculator")
    assert passed is True
    assert score == 1.0


def test_tool_accuracy_fail():
    trace = _make_trace(["search_knowledge_base"], "the answer is 42")
    score, passed, _ = metrics.tool_accuracy(trace, "calculator")
    assert passed is False
    assert score == 0.0


def test_tool_accuracy_no_expectation_skips():
    trace = _make_trace([], "some answer")
    score, passed, _ = metrics.tool_accuracy(trace, "")
    assert passed is True


def test_keyword_groundedness_pass():
    trace = _make_trace(["calculator"], "The result is 4410.")
    score, passed, _ = metrics.keyword_groundedness(trace, ["4410"])
    assert passed is True
    assert score == 1.0


def test_keyword_groundedness_fail():
    trace = _make_trace(["calculator"], "The result is 9999.")
    score, passed, _ = metrics.keyword_groundedness(trace, ["4410"])
    assert passed is False
    assert score == 0.0


def test_latency_budget_pass():
    trace = _make_trace(["calculator"], "answer", latency_ms=1000)
    score, passed, _ = metrics.latency_budget(trace, budget_ms=5000)
    assert passed is True


def test_latency_budget_fail():
    trace = _make_trace(["calculator"], "answer", latency_ms=9000)
    score, passed, _ = metrics.latency_budget(trace, budget_ms=5000)
    assert passed is False


def test_estimate_cost_nonzero_for_tokens():
    trace = _make_trace(["calculator"], "answer")
    trace.steps[0].input_tokens = 100
    trace.steps[0].output_tokens = 50
    cost = metrics.estimate_cost_usd(trace)
    assert cost > 0
