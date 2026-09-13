"""
Tests for the Comprehensive Tool Evaluation Module.
Verifies all 10 captured telemetry fields and 7 evaluation dimensions:
1. Tool selection accuracy
2. Tool argument correctness
3. Tool execution success
4. Tool efficiency
5. Unnecessary tool calls
6. Tool sequence correctness
7. Retry behavior
"""

import pytest
from src.core.entities import TestCase, Trace, Span
from src.evaluation.tool_evaluators import (
    extract_tool_telemetry,
    ToolSelectionAccuracyEvaluator,
    ToolArgumentCorrectnessEvaluator,
    ToolExecutionSuccessEvaluator,
    ToolEfficiencyEvaluator,
    UnnecessaryToolCallsEvaluator,
    ToolSequenceCorrectnessEvaluator,
    ToolRetryBehaviorEvaluator,
    ToolEvaluationSuiteEvaluator,
    CapturedToolCall,
)
from src.evaluation.evaluators import get_evaluator


def test_extract_tool_telemetry_captures_all_10_fields():
    """Verify that every tool call in a trace captures all 10 standard dimensions."""
    trace = Trace(task_id="T001", query="Search knowledge base for refund")
    trace.log_span(Span(
        step_type="tool",
        tool_name="search_knowledge_base",
        input_data='{"query": "refund policy"}',
        output_data="[Refund Policy] 7 business days",
        latency_ms=124.5,
        status="success",
        error=None,
    ))

    test_case = TestCase(
        test_id="T001",
        query="Search knowledge base for refund",
        expected_tools=["search_knowledge_base"],
        expected_arguments={"search_knowledge_base": {"query": "refund policy"}},
    )

    summary = extract_tool_telemetry(trace, test_case)
    assert len(summary.tool_calls) == 1
    call: CapturedToolCall = summary.tool_calls[0]

    # Verify all 10 captured fields
    assert call.tool_name == "search_knowledge_base"                      # 1. tool name
    assert call.expected_tool == "search_knowledge_base"                  # 2. expected tool
    assert call.arguments == {"query": "refund policy"}                   # 3. arguments
    assert call.expected_arguments == {"query": "refund policy"}          # 4. expected arguments
    assert call.execution_status == "success"                             # 5. execution status
    assert call.response == "[Refund Policy] 7 business days"             # 6. response
    assert call.latency == 124.5                                          # 7. latency
    assert call.retry_count == 0                                          # 8. retry count
    assert call.error is None                                             # 9. error
    assert call.sequence_position == 1                                    # 10. sequence position


def test_tool_selection_accuracy_wrong_tool_example():
    """Test the exact example from the requirement:
    Expected: search_knowledge_base
    Actual: web_search
    Result: FAILED
    Reason: Wrong tool selected.
    """
    trace = Trace(task_id="T001", query="Search refund policy")
    trace.log_span(Span(
        step_type="tool",
        tool_name="web_search",
        input_data='{"q": "refund"}',
        output_data="web results",
    ))

    test_case = TestCase(
        test_id="T001",
        query="Search refund policy",
        expected_tool="search_knowledge_base",
    )

    evaluator = ToolSelectionAccuracyEvaluator()
    result = evaluator.evaluate(test_case=test_case, trace=trace)

    assert result.passed is False
    assert result.score == 0.0
    assert "Wrong tool selected" in result.explanation
    assert "Expected: search_knowledge_base" in result.explanation
    assert "Actual: web_search" in result.explanation


def test_tool_selection_accuracy_forbidden_tool():
    """Verify that calling a forbidden tool immediately fails selection accuracy."""
    trace = Trace(task_id="T002", query="Calculate sum")
    trace.log_span(Span(step_type="tool", tool_name="system_exec", input_data="rm -rf"))

    test_case = TestCase(
        test_id="T002",
        query="Calculate sum",
        expected_tools=["calculator"],
        forbidden_tools=["system_exec"],
    )

    evaluator = ToolSelectionAccuracyEvaluator()
    result = evaluator.evaluate(test_case=test_case, trace=trace)
    assert result.passed is False
    assert "Forbidden tool(s) called" in result.explanation


def test_tool_argument_correctness():
    """Verify tool argument validation with valid, invalid, and expected arguments."""
    evaluator = ToolArgumentCorrectnessEvaluator()

    # 1. Valid arguments matching expectation
    trace_valid = Trace(task_id="T003", query="Compute 2+2")
    trace_valid.log_span(Span(
        step_type="tool",
        tool_name="calculator",
        input_data='{"expression": "2 + 2"}',
    ))
    tc_valid = TestCase(
        test_id="T003",
        query="Compute 2+2",
        expected_tool="calculator",
        expected_arguments={"calculator": {"expression": "2 + 2"}},
    )
    res_valid = evaluator.evaluate(test_case=tc_valid, trace=trace_valid)
    assert res_valid.passed is True
    assert res_valid.score == 1.0

    # 2. Empty / invalid arguments
    trace_empty = Trace(task_id="T003", query="Compute 2+2")
    trace_empty.log_span(Span(
        step_type="tool",
        tool_name="calculator",
        input_data="",
    ))
    res_empty = evaluator.evaluate(test_case=tc_valid, trace=trace_empty)
    assert res_empty.passed is False
    assert "argument defects" in res_empty.explanation


def test_tool_execution_success():
    """Verify tool execution success and failure diagnostics."""
    evaluator = ToolExecutionSuccessEvaluator()

    # 1. Succeeded tool call
    trace_ok = Trace(task_id="T004", query="Date check")
    trace_ok.log_span(Span(step_type="tool", tool_name="get_current_date", status="success"))
    tc = TestCase(test_id="T004", query="Date check")
    res_ok = evaluator.evaluate(test_case=tc, trace=trace_ok)
    assert res_ok.passed is True
    assert res_ok.score == 1.0

    # 2. Failed tool call with error
    trace_fail = Trace(task_id="T004", query="Date check")
    trace_fail.log_span(Span(
        step_type="tool",
        tool_name="calculator",
        status="error",
        error="DivisionByZeroError: cannot divide by zero",
    ))
    res_fail = evaluator.evaluate(test_case=tc, trace=trace_fail)
    assert res_fail.passed is False
    assert res_fail.score == 0.0
    assert "DivisionByZeroError" in res_fail.explanation


def test_unnecessary_tool_calls_detection():
    """Test the example:
    Expected: LLM → calculator → final answer
    Actual: LLM → search → search → calculator → LLM → final answer
    Flag: ⚠️ 2 potentially unnecessary tool calls
    """
    trace = Trace(task_id="T005", query="What is 50 * 2?")
    # Actual steps: LLM -> search -> search -> calculator -> LLM -> final answer
    trace.log_span(Span(step_type="llm", operation_name="LLM Plan"))
    trace.log_span(Span(step_type="tool", tool_name="search_knowledge_base", input_data="50 * 2"))
    trace.log_span(Span(step_type="tool", tool_name="search_knowledge_base", input_data="50 * 2"))
    trace.log_span(Span(step_type="tool", tool_name="calculator", input_data="50 * 2"))
    trace.log_span(Span(step_type="llm", operation_name="LLM Generate"))
    trace.finish("100")

    test_case = TestCase(
        test_id="T005",
        query="What is 50 * 2?",
        expected_tools=["calculator"],
        expected_tool_sequence=["calculator"],
    )

    evaluator = UnnecessaryToolCallsEvaluator()
    result = evaluator.evaluate(test_case=test_case, trace=trace)

    assert result.passed is False
    assert result.evidence["unnecessary_count"] == 2
    assert "⚠️ 2 potentially unnecessary tool calls detected" in result.explanation
    assert len(result.evidence["flags"]) == 2


def test_tool_sequence_correctness():
    """Verify tool sequence evaluation with exact and out-of-order calls."""
    evaluator = ToolSequenceCorrectnessEvaluator()

    # 1. Correct sequence: search -> calculator
    trace_correct = Trace(task_id="T006", query="Find price and compute discount")
    trace_correct.log_span(Span(step_type="tool", tool_name="search_knowledge_base"))
    trace_correct.log_span(Span(step_type="tool", tool_name="calculator"))

    tc = TestCase(
        test_id="T006",
        query="Find price and compute discount",
        expected_tool_sequence=["search_knowledge_base", "calculator"],
    )
    res_correct = evaluator.evaluate(test_case=tc, trace=trace_correct)
    assert res_correct.passed is True
    assert res_correct.score == 1.0

    # 2. Inverted sequence: calculator -> search
    trace_inverted = Trace(task_id="T006", query="Find price and compute discount")
    trace_inverted.log_span(Span(step_type="tool", tool_name="calculator"))
    trace_inverted.log_span(Span(step_type="tool", tool_name="search_knowledge_base"))

    res_inverted = evaluator.evaluate(test_case=tc, trace=trace_inverted)
    assert res_inverted.passed is False
    assert "Tool sequence mismatch" in res_inverted.explanation


def test_retry_behavior_evaluator():
    """Verify detection of excessive consecutive duplicate retries."""
    evaluator = ToolRetryBehaviorEvaluator(max_allowed_retries=1)

    # 1. Normal single retry
    trace_normal = Trace(task_id="T007", query="Query")
    trace_normal.log_span(Span(step_type="tool", tool_name="calculator", input_data="1+1"))
    trace_normal.log_span(Span(step_type="tool", tool_name="calculator", input_data="1+1"))
    tc = TestCase(test_id="T007", query="Query")
    res_normal = evaluator.evaluate(test_case=tc, trace=trace_normal)
    assert res_normal.passed is True

    # 2. Retry loop with 4 identical consecutive calls
    trace_loop = Trace(task_id="T007", query="Query")
    for _ in range(4):
        trace_loop.log_span(Span(step_type="tool", tool_name="calculator", input_data="1+1"))
    res_loop = evaluator.evaluate(test_case=tc, trace=trace_loop)
    assert res_loop.passed is False
    assert "Detected excessive retry loop" in res_loop.explanation


def test_tool_efficiency_evaluator():
    """Verify tool efficiency scoring against productive calls."""
    evaluator = ToolEfficiencyEvaluator(threshold=0.8)

    # 1. Optimal trace (1 expected, 1 called)
    trace_opt = Trace(task_id="T008", query="Query")
    trace_opt.log_span(Span(step_type="tool", tool_name="calculator", input_data="10*10"))
    tc = TestCase(test_id="T008", query="Query", expected_tools=["calculator"])
    res_opt = evaluator.evaluate(test_case=tc, trace=trace_opt)
    assert res_opt.passed is True
    assert res_opt.score == 1.0

    # 2. Inefficient trace with unnecessary calls
    trace_ineff = Trace(task_id="T008", query="Query")
    trace_ineff.log_span(Span(step_type="tool", tool_name="web_search", input_data="test"))
    trace_ineff.log_span(Span(step_type="tool", tool_name="web_search", input_data="test2"))
    trace_ineff.log_span(Span(step_type="tool", tool_name="calculator", input_data="10*10"))
    res_ineff = evaluator.evaluate(test_case=tc, trace=trace_ineff)
    assert res_ineff.passed is False
    assert res_ineff.score < 0.8


def test_tool_evaluation_suite_composite():
    """Verify composite suite evaluator running all 7 dimensions together."""
    suite = ToolEvaluationSuiteEvaluator()

    trace = Trace(task_id="T009", query="What's the date?")
    trace.log_span(Span(
        step_type="tool",
        tool_name="get_current_date",
        input_data="{}",
        output_data="Friday, 2026-09-04",
        status="success",
        latency_ms=25.0,
    ))
    trace.finish("Today is Friday, 2026-09-04")

    tc = TestCase(
        test_id="T009",
        query="What's the date?",
        expected_tool="get_current_date",
        expected_tool_sequence=["get_current_date"],
    )

    res = suite.evaluate(test_case=tc, trace=trace)
    assert res.passed is True
    assert res.score >= 0.9
    assert "tool_selection_accuracy" in res.evidence["dimension_breakdown"]
    assert "unnecessary_tool_calls" in res.evidence["dimension_breakdown"]


def test_evaluator_registry_instantiation():
    """Verify all tool evaluators can be instantiated via get_evaluator."""
    for name in [
        "tool_selection_accuracy",
        "tool_argument_correctness",
        "tool_execution_success",
        "tool_efficiency",
        "unnecessary_tool_calls",
        "tool_sequence_correctness",
        "tool_retry_behavior",
        "tool_evaluation_suite",
    ]:
        ev = get_evaluator(name)
        assert ev is not None
