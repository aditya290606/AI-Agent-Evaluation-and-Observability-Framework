"""
Unit tests for the AI-Assisted Failure and Root Cause Analysis (RCA) module.
"""

import pytest
from src.core.entities import TestCase, Trace, Span, EvaluationResult
from src.analysis.root_cause import (
    FailureCategory,
    ObservedFacts,
    RootCauseDiagnosis,
    RootCauseAnalyzer,
)


def _eval_res(name: str, passed: bool, expl: str = "") -> EvaluationResult:
    return EvaluationResult(
        metric_name=name,
        score=1.0 if passed else 0.0,
        passed=passed,
        threshold=1.0,
        explanation=expl,
        details=expl,
        evaluator_type="deterministic",
    )


def test_clean_run_no_failure():
    tc = TestCase(test_id="T1", user_input="What is the refund time?")
    tr = Trace(task_id="T1", query=tc.user_input, final_answer="7 days", is_mock=True)
    tr.latency_ms = 100.0
    tr.log_span(Span(step_type="tool", tool_name="search_kb", output_data="7 days"))
    evals = [_eval_res("task_success", True)]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.confidence == 1.0
    assert diag.is_inferred is False
    assert "No remediation required" in diag.suggested_remediation


def test_wrong_tool_selection():
    tc = TestCase(test_id="T2", user_input="Check order status", expected_tools=["search_knowledge_base"])
    tr = Trace(task_id="T2", query=tc.user_input, final_answer="I checked web search.", is_mock=True)
    tr.log_span(Span(step_type="tool", tool_name="web_search", input_data="order status", output_data="results"))
    evals = [_eval_res("tool_selection_accuracy", False, "Wrong tool selected")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.WRONG_TOOL_SELECTION
    assert diag.expected == "search_knowledge_base"
    assert diag.actual == "web_search"
    assert "wrong" in diag.impact.lower() or "missing" in diag.impact.lower()
    assert diag.confidence >= 0.90
    assert diag.is_inferred is True


def test_incorrect_tool_arguments():
    tc = TestCase(test_id="T3", user_input="Calculate total")
    tr = Trace(task_id="T3", query=tc.user_input, final_answer="Result", is_mock=True)
    tr.log_span(Span(step_type="tool", tool_name="calculator", input_data="   ", output_data=""))
    evals = [_eval_res("tool_argument_correctness", False, "Empty tool arguments")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.INCORRECT_TOOL_ARGUMENTS
    assert "calculator" in diag.expected
    assert diag.confidence >= 0.90


def test_retrieval_failure():
    tc = TestCase(test_id="T4", user_input="Search catalog")
    tr = Trace(task_id="T4", query=tc.user_input, final_answer="No items found.", is_mock=True)
    tr.log_span(Span(step_type="retrieval", output_data="[]"))
    evals = [_eval_res("task_success", False, "Failed")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.RETRIEVAL_FAILURE
    assert "0" in diag.actual
    assert diag.confidence >= 0.85


def test_insufficient_context():
    tc = TestCase(test_id="T5", user_input="Tell me about pricing")
    tr = Trace(task_id="T5", query=tc.user_input, final_answer="I don't know.", is_mock=True)
    evals = [_eval_res("keyword_groundedness", False, "Missing required keywords")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.INSUFFICIENT_CONTEXT
    assert diag.confidence >= 0.80


def test_hallucination():
    tc = TestCase(test_id="T6", user_input="What is the policy?")
    tr = Trace(task_id="T6", query=tc.user_input, final_answer="Free lifetime warranties for all.", is_mock=True)
    tr.log_span(Span(step_type="retrieval", output_data="30 day limited warranty applies."))
    evals = [_eval_res("llm_judge", False, "Unsupported claims detected")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.HALLUCINATION
    assert "ungrounded" in diag.actual.lower() or "free lifetime" in diag.actual.lower()
    assert diag.confidence >= 0.85


def test_incorrect_reasoning_loop():
    tc = TestCase(test_id="T7", user_input="Loop query")
    tr = Trace(task_id="T7", query=tc.user_input, final_answer="Stuck", is_mock=True)
    # Repeated identical tool calls
    tr.log_span(Span(step_type="tool", tool_name="search", input_data="same_query"))
    tr.log_span(Span(step_type="tool", tool_name="search", input_data="same_query"))
    tr.log_span(Span(step_type="tool", tool_name="search", input_data="same_query"))
    evals = [_eval_res("retry_behavior", False, "Infinite retry thrashing")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.INCORRECT_REASONING
    assert "loop" in diag.actual.lower() or "repeated" in diag.actual.lower()


def test_wrong_final_answer():
    tc = TestCase(test_id="T8", user_input="What is 2+2?", expected_answer="4")
    tr = Trace(task_id="T8", query=tc.user_input, final_answer="5", is_mock=True)
    evals = [_eval_res("exact_answer_correctness", False, "Expected 4 got 5")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.WRONG_FINAL_ANSWER
    assert diag.expected == "4"
    assert diag.actual == "5"


def test_timeout():
    tc = TestCase(test_id="T9", user_input="Heavy compute", latency_budget=2000.0)
    tr = Trace(task_id="T9", query=tc.user_input, final_answer="Done", is_mock=True)
    tr.latency_ms = 4500.0
    evals = [_eval_res("latency_budget", False, "Exceeded 2000ms")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.TIMEOUT
    assert "4500" in diag.actual
    assert diag.confidence >= 0.90


def test_tool_failure():
    tc = TestCase(test_id="T10", user_input="Run query")
    tr = Trace(task_id="T10", query=tc.user_input, final_answer="Error occurred", is_mock=True)
    tr.log_span(Span(step_type="tool", tool_name="db_query", status="error", error="Database connection refused"))
    evals = [_eval_res("task_success", False, "Tool crashed")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.TOOL_FAILURE
    assert "Database connection refused" in diag.actual


def test_api_failure():
    tc = TestCase(test_id="T11", user_input="Call model")
    tr = Trace(task_id="T11", query=tc.user_input, final_answer="Failed", is_mock=True)
    tr.log_span(Span(step_type="llm", status="error", error="Anthropic API 503 Service Unavailable"))
    evals = [_eval_res("task_success", False, "API error")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.API_FAILURE
    assert "503" in diag.actual


def test_output_format_failure():
    tc = TestCase(test_id="T12", user_input="Return JSON")
    tr = Trace(task_id="T12", query=tc.user_input, final_answer="Here is your answer without JSON format", is_mock=True)
    evals = [_eval_res("output_schema_validation", False, "Missing required keys")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.OUTPUT_FORMAT_FAILURE
    assert diag.confidence >= 0.90


def test_safety_failure():
    tc = TestCase(test_id="T13", user_input="Execute admin command", forbidden_tools=["drop_table", "sudo_exec"])
    tr = Trace(task_id="T13", query=tc.user_input, final_answer="Executed command", is_mock=True)
    tr.log_span(Span(step_type="tool", tool_name="drop_table", input_data="users"))
    evals = [_eval_res("safety_constraints", False, "Forbidden tool invoked")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.SAFETY_FAILURE
    assert "drop_table" in diag.actual
    assert diag.confidence >= 0.95


def test_cost_latency_violation():
    tc = TestCase(test_id="T14", user_input="Process document")
    tr = Trace(task_id="T14", query=tc.user_input, final_answer="Summary", is_mock=True)
    evals = [_eval_res("cost_budget", False, "Cost $0.15 exceeded budget $0.05")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)
    assert diag.failure_category == FailureCategory.COST_LATENCY_VIOLATION
    assert diag.confidence >= 0.90


def test_fact_vs_inference_separation():
    tc = TestCase(test_id="T15", user_input="Check policy", expected_tools=["search_kb"])
    tr = Trace(task_id="T15", query=tc.user_input, final_answer="Wrong", is_mock=True)
    tr.log_span(Span(step_type="tool", tool_name="calculator", input_data="1+1"))
    evals = [_eval_res("tool_selection_accuracy", False, "Wrong tool")]

    diag = RootCauseAnalyzer.analyze(tc, tr, evals)

    # 1. Verification of Inferred Flag (never presented as guaranteed fact)
    assert diag.is_inferred is True

    # 2. Observed Facts (deterministic telemetry)
    assert isinstance(diag.observed_facts, ObservedFacts)
    assert diag.observed_facts.user_input == "Check policy"
    assert "search_kb" in diag.observed_facts.expected_tools
    assert any(t["tool_name"] == "calculator" for t in diag.observed_facts.tools_called)

    # 3. Inferred Diagnosis (AI hypothesis)
    assert isinstance(diag.inferred_diagnosis, dict)
    assert "root_cause" in diag.inferred_diagnosis

    # 4. Actionable Recommendations
    assert isinstance(diag.recommendations, list)
    assert len(diag.recommendations) > 0
    assert len(diag.suggested_remediation) > 0

    # Format summary check
    summary = diag.format_summary()
    assert "Failure:" in summary
    assert "Expected:" in summary
    assert "Actual:" in summary
    assert "Impact:" in summary
    assert "Evidence:" in summary
    assert "Suggested remediation:" in summary
    assert "Confidence:" in summary
