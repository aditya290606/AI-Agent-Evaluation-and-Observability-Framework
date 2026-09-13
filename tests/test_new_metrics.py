"""
Unit tests for the 9 Evaluation Metric dimensions.
"""

import pytest
from src.core.entities import TestCase, Trace, Span
from src.evaluation.metrics import (
    ExactAnswerMetric,
    SemanticAnswerMetric,
    ToolSelectionMetric,
    ToolArgumentMetric,
    ExpectedBehaviorMetric,
    GroundednessMetric,
    LatencyMetric,
    StructuredOutputMetric,
    SafetyConstraintMetric,
)


def make_trace(answer="7 business days", tools=None, latency=150.0, input_tokens=100, output_tokens=50):
    tools = tools or ["search_knowledge_base"]
    per_span_lat = latency / len(tools) if tools else latency
    spans = [
        Span(
            step_index=i,
            step_type="tool_call",
            tool_name=t,
            input_data='{"query": "refund"}',
            output_data="7 days",
            latency_ms=per_span_lat,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        for i, t in enumerate(tools)
    ]
    return Trace(
        task_id="T001",
        query="How many days for refund?",
        final_answer=answer,
        spans=spans,
    )


def test_exact_answer_metric():
    metric = ExactAnswerMetric()
    tc = TestCase(test_id="T1", user_input="Days?", expected_answer="7 business days")
    trace_pass = make_trace(answer="7 business days")
    res_pass = metric.evaluate(trace_pass, tc)
    assert res_pass.passed is True
    assert res_pass.score == 1.0

    trace_fail = make_trace(answer="24 hours")
    res_fail = metric.evaluate(trace_fail, tc)
    assert res_fail.passed is False
    assert res_fail.score == 0.0


def test_semantic_answer_metric():
    metric = SemanticAnswerMetric()
    tc = TestCase(test_id="T1", user_input="Days?", expected_answer="7 business days")
    trace = make_trace(answer="It takes about 7 business days for the refund.")
    res = metric.evaluate(trace, tc)
    assert res.passed is True
    assert res.score >= 0.5


def test_tool_selection_metric():
    metric = ToolSelectionMetric()
    tc = TestCase(
        test_id="T1",
        user_input="Search",
        expected_tools=["search_knowledge_base"],
        forbidden_tools=["execute_sql", "delete_user"],
    )

    # Pass: expected tool called, forbidden not called
    trace_pass = make_trace(tools=["search_knowledge_base"])
    assert metric.evaluate(trace_pass, tc).passed is True

    # Fail: forbidden tool called
    trace_fail_forb = make_trace(tools=["search_knowledge_base", "execute_sql"])
    res_forb = metric.evaluate(trace_fail_forb, tc)
    assert res_forb.passed is False
    assert "forbidden" in res_forb.details.lower()

    # Fail: expected tool missing
    trace_fail_missing = make_trace(tools=["calculator"])
    res_missing = metric.evaluate(trace_fail_missing, tc)
    assert res_missing.passed is False


def test_tool_arguments_metric():
    metric = ToolArgumentMetric()
    tc = TestCase(test_id="T1", user_input="Search")
    trace = make_trace()
    res = metric.evaluate(trace, tc)
    assert res.passed is True
    assert res.score == 1.0


def test_expected_behavior_metric():
    metric = ExpectedBehaviorMetric()
    tc = TestCase(test_id="T1", user_input="Search", expected_behavior="Must query docs and return answer")
    trace = make_trace(answer="Refund takes 7 business days.")
    res = metric.evaluate(trace, tc)
    assert res.passed is True


def test_groundedness_metric():
    metric = GroundednessMetric()
    tc = TestCase(test_id="T1", user_input="Days?", expected_keywords=["7", "business days"])
    trace = make_trace(answer="Refund processed in 7 business days.")
    res = metric.evaluate(trace, tc)
    assert res.passed is True


def test_latency_metric():
    metric = LatencyMetric()
    tc = TestCase(test_id="T1", user_input="Days?", latency_budget=500.0)
    trace_fast = make_trace(latency=120.0)
    assert metric.evaluate(trace_fast, tc).passed is True

    trace_slow = make_trace(latency=1500.0)
    assert metric.evaluate(trace_slow, tc).passed is False


def test_structured_output_metric():
    metric = StructuredOutputMetric()
    schema = {"type": "object", "required": ["refund_days", "status"]}
    tc = TestCase(test_id="T1", user_input="JSON please", expected_output_schema=schema)

    trace_valid = make_trace(answer='{"refund_days": 7, "status": "approved"}')
    res_valid = metric.evaluate(trace_valid, tc)
    assert res_valid.passed is True

    trace_missing_key = make_trace(answer='{"refund_days": 7}')
    res_missing = metric.evaluate(trace_missing_key, tc)
    assert res_missing.passed is False

    trace_bad_json = make_trace(answer="Not valid json")
    res_bad = metric.evaluate(trace_bad_json, tc)
    assert res_bad.passed is False


def test_safety_constraint_metric():
    metric = SafetyConstraintMetric()
    tc = TestCase(test_id="T1", user_input="Safe query", forbidden_tools=["drop_table"])

    trace_safe = make_trace(answer="Clean text without sensitive data.")
    assert metric.evaluate(trace_safe, tc).passed is True

    trace_forbidden = make_trace(tools=["drop_table"])
    assert metric.evaluate(trace_forbidden, tc).passed is False
