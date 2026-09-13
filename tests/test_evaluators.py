"""
Comprehensive unit tests for all 14 modular evaluators in the evaluation engine.
"""

import pytest
from src.core.entities import TestCase, Trace, Span
from src.evaluation.evaluators import (
    TaskSuccessEvaluator,
    ExactAnswerEvaluator,
    SemanticAnswerEvaluator,
    ToolSelectionEvaluator,
    ToolArgumentEvaluator,
    KeywordGroundednessEvaluator,
    ContextGroundednessEvaluator,
    LLMJudgeEvaluator,
    LatencyBudgetEvaluator,
    TokenUsageEvaluator,
    CostBudgetEvaluator,
    OutputSchemaEvaluator,
    ErrorRateEvaluator,
    RetryBehaviorEvaluator,
    get_evaluator,
)


def _make_trace(final_answer="Refunds take 7 business days.", spans=None, latency_ms=100.0, in_tok=50, out_tok=25):
    t = Trace(task_id="T001", query="Refund time?", final_answer=final_answer, is_mock=True)
    t.latency_ms = latency_ms
    if spans:
        for s in spans:
            t.log_span(s)
    else:
        t.log_span(Span(step_type="tool", tool_name="search_knowledge_base", input_data="refund policy", output_data="Refunds take 7 business days."))
        t.log_span(Span(step_type="final_answer", output_data=final_answer, input_tokens=in_tok, output_tokens=out_tok))
    return t


def test_task_success_evaluator():
    ev = TaskSuccessEvaluator()
    tc = TestCase(test_id="T1", user_input="Help", expected_keywords=["7 days"])
    t_pass = _make_trace("Your refund will arrive in 7 days.")
    res_pass = ev(test_case=tc, trace=t_pass)
    assert res_pass.passed is True
    assert res_pass.score == 1.0
    assert "explanation" in res_pass.to_dict()

    t_empty = _make_trace("")
    res_empty = ev(test_case=tc, trace=t_empty)
    assert res_empty.passed is False
    assert res_empty.score == 0.0


def test_exact_answer_evaluator():
    ev = ExactAnswerEvaluator()
    tc = TestCase(test_id="T2", user_input="Math", expected_answer="4410")
    t_exact = _make_trace("The answer is 4410.")
    res = ev(test_case=tc, trace=t_exact)
    assert res.passed is True
    assert res.score == 1.0
    assert res.evidence["expected"] == "4410"

    t_wrong = _make_trace("The answer is 5000.")
    res_wrong = ev(test_case=tc, trace=t_wrong)
    assert res_wrong.passed is False
    assert res_wrong.score == 0.0


def test_semantic_answer_evaluator():
    ev = SemanticAnswerEvaluator(threshold=0.6)
    tc = TestCase(test_id="T3", user_input="Policy", expected_answer="refunds are processed within seven business days")
    t_sem = _make_trace("Refunds are issued within 7 business days of receipt.")
    res = ev(test_case=tc, trace=t_sem)
    assert res.score > 0.4
    assert res.evaluator_type == "semantic"


def test_tool_selection_evaluator():
    ev = ToolSelectionEvaluator()
    tc = TestCase(test_id="T4", user_input="Lookup", expected_tools=["calculator"], forbidden_tools=["delete_db"])
    t_ok = _make_trace(spans=[Span(step_type="tool", tool_name="calculator", input_data="2+2")])
    res_ok = ev(test_case=tc, trace=t_ok)
    assert res_ok.passed is True

    t_forbid = _make_trace(spans=[
        Span(step_type="tool", tool_name="calculator", input_data="2+2"),
        Span(step_type="tool", tool_name="delete_db", input_data="DROP TABLE"),
    ])
    res_forbid = ev(test_case=tc, trace=t_forbid)
    assert res_forbid.passed is False
    assert "forbidden" in res_forbid.explanation.lower()


def test_tool_argument_evaluator():
    ev = ToolArgumentEvaluator()
    tc = TestCase(test_id="T5", user_input="Run calc")
    t_good = _make_trace(spans=[Span(step_type="tool", tool_name="calculator", input_data="100 * 5")])
    res_good = ev(test_case=tc, trace=t_good)
    assert res_good.passed is True

    t_bad = _make_trace(spans=[Span(step_type="tool", tool_name="calculator", input_data="")])
    res_bad = ev(test_case=tc, trace=t_bad)
    assert res_bad.passed is False


def test_keyword_groundedness_evaluator():
    ev = KeywordGroundednessEvaluator()
    tc = TestCase(test_id="T6", user_input="Days", expected_keywords=["7 business days", "original payment"])
    t = _make_trace("Refunds take 7 business days to the original payment method.")
    res = ev(test_case=tc, trace=t)
    assert res.passed is True
    assert res.score == 1.0


def test_context_groundedness_evaluator():
    ev = ContextGroundednessEvaluator(threshold=0.3)
    tc = TestCase(test_id="T7", user_input="Policy")
    spans = [
        Span(step_type="retrieval", output_data="Pro plan costs 1499 INR per month."),
        Span(step_type="final_answer", output_data="The Pro plan is 1499 INR each month."),
    ]
    t = _make_trace(final_answer="The Pro plan is 1499 INR each month.", spans=spans)
    res = ev(test_case=tc, trace=t)
    assert res.score >= 0.3
    assert res.passed is True


def test_llm_judge_evaluator():
    ev = LLMJudgeEvaluator()
    tc = TestCase(test_id="T8", user_input="Date question")
    t = _make_trace("Today is Thursday.")
    res = ev(test_case=tc, trace=t)
    assert res.score >= 0.0
    assert res.evaluator_type in ["LLM Judge — MOCK", "LLM Judge — LIVE"]


def test_latency_budget_evaluator():
    ev = LatencyBudgetEvaluator()
    tc = TestCase(test_id="T9", user_input="Fast query", latency_budget=500.0)
    t_fast = _make_trace(latency_ms=120.0)
    res_fast = ev(test_case=tc, trace=t_fast)
    assert res_fast.passed is True
    assert res_fast.score == 1.0

    t_slow = _make_trace(latency_ms=1200.0)
    res_slow = ev(test_case=tc, trace=t_slow)
    assert res_slow.passed is False
    assert res_slow.score < 1.0


def test_token_usage_evaluator():
    ev = TokenUsageEvaluator(token_budget=1000)
    tc = TestCase(test_id="T10", user_input="Tokens")
    t_light = _make_trace(spans=[Span(step_type="final_answer", input_tokens=200, output_tokens=100)])
    res_light = ev(test_case=tc, trace=t_light)
    assert res_light.passed is True

    t_heavy = _make_trace(spans=[Span(step_type="final_answer", input_tokens=800, output_tokens=600)])
    res_heavy = ev(test_case=tc, trace=t_heavy)
    assert res_heavy.passed is False
    assert res_heavy.score < 1.0


def test_cost_budget_evaluator():
    ev = CostBudgetEvaluator(cost_budget_usd=0.01)
    tc = TestCase(test_id="T11", user_input="Cost")
    t_cheap = _make_trace(spans=[Span(step_type="final_answer", input_tokens=100, output_tokens=50)])
    res_cheap = ev(test_case=tc, trace=t_cheap)
    assert res_cheap.passed is True


def test_output_schema_evaluator():
    ev = OutputSchemaEvaluator()
    tc = TestCase(test_id="T12", user_input="JSON", expected_output_schema={"type": "object", "required": ["result", "status"]})
    t_valid = _make_trace('{"result": 42, "status": "ok"}')
    res_valid = ev(test_case=tc, trace=t_valid)
    assert res_valid.passed is True

    t_invalid = _make_trace('{"result": 42}')
    res_invalid = ev(test_case=tc, trace=t_invalid)
    assert res_invalid.passed is False
    assert "missing" in res_invalid.explanation.lower()


def test_error_rate_evaluator():
    ev = ErrorRateEvaluator()
    tc = TestCase(test_id="T13", user_input="Error check")
    t_clean = _make_trace(spans=[
        Span(step_type="tool", status="success"),
        Span(step_type="final_answer", status="success"),
    ])
    assert ev(test_case=tc, trace=t_clean).passed is True

    t_err = _make_trace(spans=[
        Span(step_type="tool", status="error", error="Database connection failed"),
        Span(step_type="final_answer", status="success"),
    ])
    res_err = ev(test_case=tc, trace=t_err)
    assert res_err.passed is False
    assert res_err.score < 1.0


def test_retry_behavior_evaluator():
    ev = RetryBehaviorEvaluator(max_allowed_retries=2)
    tc = TestCase(test_id="T14", user_input="Loop check")
    t_loop = _make_trace(spans=[
        Span(step_type="tool", tool_name="search", input_data="same_query"),
        Span(step_type="tool", tool_name="search", input_data="same_query"),
        Span(step_type="tool", tool_name="search", input_data="same_query"),
        Span(step_type="tool", tool_name="search", input_data="same_query"),
    ])
    res_loop = ev(test_case=tc, trace=t_loop)
    assert res_loop.passed is False
    assert "retry loop" in res_loop.explanation.lower()


def test_get_evaluator_factory():
    evaluator = get_evaluator("task_success")
    assert isinstance(evaluator, TaskSuccessEvaluator)
    assert evaluator.name == "task_success"
