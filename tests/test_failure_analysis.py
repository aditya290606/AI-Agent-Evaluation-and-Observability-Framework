"""
Unit tests for the Failure Analysis module (src/analysis/failure_analysis.py).

Verifies:
1. Support for all 6 required failure categories:
   - Wrong Tool
   - Incorrect Tool Arguments
   - Incorrect Answer
   - Poor Grounding
   - Latency Violation
   - Tool/API Failure
2. Output format:
   - Failure (failure_type)
   - Expected
   - Actual
   - Evidence
   - Likely Root Cause
   - Recommendation
3. Separation of observed evidence from inferred diagnosis:
   - Evidence is factual and traces back to specific spans/metrics
   - Root cause is clearly marked as inferred/hypothesis
   - is_inferred flag is set
4. Edge cases:
   - Clean/passing evaluations (NO_FAILURE)
   - Multiple failures detected via analyze_all
"""

import pytest
from src.analysis.failure_analysis import (
    FailureAnalyzer,
    FailureType,
    FailureAnalysisResult,
)
from src.core.entities import TestCase, Trace, Span, EvaluationResult


class MockSpan:
    """Helper to mock a trace span."""
    def __init__(
        self,
        span_type="tool",
        tool_name="web_search",
        input_data="",
        output_data="",
        status="success",
        error=None,
        latency_ms=100.0,
    ):
        self.span_type = span_type
        self.tool_name = tool_name
        self.input_data = input_data
        self.output_data = output_data
        self.status = status
        self.error = error
        self.latency_ms = latency_ms


class TestFailureAnalyzerCategories:
    """Tests each of the 6 failure categories."""

    def test_wrong_tool_selected(self):
        """User example:
        Expected: search_knowledge_base
        Actual: web_search
        Evidence: Trace span #1
        Root Cause: [INFERRED] ...
        Recommendation: Review tool routing ...
        """
        tc = TestCase(
            task_id="TC_001",
            name="Policy Lookup",
            user_input="What is the return policy?",
            expected_behavior="Search policy in knowledge base",
            expected_tools=["search_knowledge_base"],
        )
        span = MockSpan(
            span_type="tool",
            tool_name="web_search",
            input_data='{"query": "return policy"}',
        )
        trace = Trace(
            task_id="TC_001",
            query="What is the return policy?",
            final_answer="Returns are accepted within 30 days.",
            spans=[span],
        )
        eval_results = [
            EvaluationResult(
                metric_name="tool_accuracy",
                score=0.0,
                passed=False,
                threshold=1.0,
                explanation="Expected tool search_knowledge_base, but web_search was called",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.WRONG_TOOL
        assert "search_knowledge_base" in result.expected
        assert "web_search" in result.actual
        assert "Trace span #1" in result.evidence
        assert "[INFERRED]" in result.root_cause
        assert "routing" in result.recommendation.lower()
        assert result.is_inferred is True
        assert result.confidence > 0.8

    def test_incorrect_tool_arguments(self):
        """Tool called was correct, but passed arguments were incorrect."""
        tc = TestCase(
            task_id="TC_002",
            name="Calculate Discount",
            user_input="Calculate 20% off $150",
            expected_behavior="Call calculator with 150 * 0.2",
            expected_tools=["calculator"],
            expected_tool_arguments={"expression": "150 * 0.2"},
        )
        span = MockSpan(
            span_type="tool",
            tool_name="calculator",
            input_data='{"expression": "150 * 0.5"}',  # Wrong argument
        )
        trace = Trace(
            task_id="TC_002",
            query="Calculate 20% off $150",
            final_answer="$75",
            spans=[span],
        )
        eval_results = [
            EvaluationResult(
                metric_name="task_success",
                score=0.0,
                passed=False,
                threshold=1.0,
                explanation="Calculation result incorrect",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.INCORRECT_TOOL_ARGS
        assert "150 * 0.2" in result.expected
        assert "150 * 0.5" in result.actual
        assert "Trace span #1" in result.evidence
        assert "[INFERRED]" in result.root_cause
        assert result.is_inferred is True

    def test_incorrect_answer(self):
        """Final answer did not match expected output."""
        tc = TestCase(
            task_id="TC_003",
            name="Capital of France",
            user_input="What is the capital of France?",
            expected_behavior="Return Paris",
            expected_answer="Paris",
        )
        trace = Trace(
            task_id="TC_003",
            query="What is the capital of France?",
            final_answer="The capital of France is London.",
            spans=[],
        )
        eval_results = [
            EvaluationResult(
                metric_name="answer_correctness",
                score=0.0,
                passed=False,
                threshold=0.8,
                explanation="Expected 'Paris', got 'The capital of France is London.'",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.INCORRECT_ANSWER
        assert "Paris" in result.expected
        assert "London" in result.actual
        assert "[INFERRED]" in result.root_cause
        assert result.is_inferred is True

    def test_poor_grounding(self):
        """Answer is not grounded in retrieved evidence/context."""
        tc = TestCase(
            task_id="TC_004",
            name="Product Availability",
            user_input="Is Product X in stock?",
            expected_behavior="Ground answer in inventory database",
        )
        trace = Trace(
            task_id="TC_004",
            query="Is Product X in stock?",
            final_answer="Product X is fully stocked with 500 units available right now.",
            spans=[],
        )
        eval_results = [
            EvaluationResult(
                metric_name="groundedness",
                score=0.2,
                passed=False,
                threshold=0.7,
                explanation="Answer makes claims about inventory not present in retrieved context.",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.POOR_GROUNDING
        assert "grounded" in result.expected.lower()
        assert "[INFERRED]" in result.root_cause
        assert "hallucinated" in result.root_cause.lower() or "retrieved" in result.root_cause.lower()
        assert result.is_inferred is True

    def test_latency_violation(self):
        """Execution duration exceeded test budget."""
        tc = TestCase(
            task_id="TC_005",
            name="Fast Lookup",
            user_input="Quick status check",
            expected_behavior="Respond within 2000ms",
            latency_budget=2000.0,
        )
        span = MockSpan(
            span_type="llm",
            tool_name=None,
            latency_ms=3500.0,
        )
        trace = Trace(
            task_id="TC_005",
            query="Quick status check",
            final_answer="All systems operational",
            spans=[span],
        )
        trace.latency_ms = 3500.0

        eval_results = [
            EvaluationResult(
                metric_name="latency",
                score=0.0,
                passed=False,
                threshold=1.0,
                explanation="Latency 3500ms exceeded budget 2000ms",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.LATENCY_VIOLATION
        assert "2000" in result.expected
        assert "3500" in result.actual
        assert "[INFERRED]" in result.root_cause
        assert result.is_inferred is True

    def test_tool_api_failure(self):
        """Tool execution crashed or returned an error status."""
        tc = TestCase(
            task_id="TC_006",
            name="Weather API Check",
            user_input="What's the weather in Tokyo?",
            expected_behavior="Call weather API",
            expected_tools=["get_weather"],
        )
        span = MockSpan(
            span_type="tool",
            tool_name="get_weather",
            status="error",
            error="ConnectionError: 503 Service Unavailable",
        )
        trace = Trace(
            task_id="TC_006",
            query="What's the weather in Tokyo?",
            final_answer="Sorry, weather service is currently down.",
            spans=[span],
        )
        eval_results = [
            EvaluationResult(
                metric_name="task_success",
                score=0.0,
                passed=False,
                threshold=1.0,
                explanation="Tool raised exception: ConnectionError",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)

        assert result.failure_type == FailureType.TOOL_API_FAILURE
        assert "503" in result.actual or "ConnectionError" in result.actual
        assert "Trace span #1" in result.evidence
        assert "[INFERRED]" in result.root_cause
        assert result.confidence >= 0.9


class TestFailureAnalyzerEdgeCases:
    """Tests clean runs, multi-failure extraction, and serialization."""

    def test_no_failure_when_all_pass(self):
        """When all metrics pass and trace has no errors, returns NO_FAILURE."""
        tc = TestCase(
            task_id="TC_CLEAN",
            name="Clean Pass",
            user_input="Hello",
            expected_behavior="Greet user",
        )
        trace = Trace(
            task_id="TC_CLEAN",
            query="Hello",
            final_answer="Hello! How can I help you today?",
            spans=[MockSpan(status="success", latency_ms=150.0)],
        )
        trace.latency_ms = 150.0

        eval_results = [
            EvaluationResult(
                metric_name="task_success",
                score=1.0,
                passed=True,
                threshold=1.0,
                explanation="All criteria met",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)
        assert result.failure_type == FailureType.NO_FAILURE

    def test_analyze_all_returns_multiple_failures(self):
        """When run has both wrong tool and latency violation, analyze_all returns both."""
        tc = TestCase(
            task_id="TC_MULTI",
            name="Multi Failure",
            user_input="Calculate tax",
            expected_tools=["calculator"],
            latency_budget=1000.0,
        )
        span = MockSpan(
            span_type="tool",
            tool_name="web_search",
            latency_ms=2500.0,
        )
        trace = Trace(
            task_id="TC_MULTI",
            query="Calculate tax",
            final_answer="Tax is $20",
            spans=[span],
        )
        trace.latency_ms = 2500.0

        eval_results = [
            EvaluationResult(metric_name="tool_accuracy", score=0.0, passed=False),
            EvaluationResult(metric_name="latency", score=0.0, passed=False),
        ]

        analyses = FailureAnalyzer.analyze_all(tc, trace, eval_results)
        failure_types = [a.failure_type for a in analyses]

        assert FailureType.WRONG_TOOL in failure_types
        assert FailureType.LATENCY_VIOLATION in failure_types

    def test_to_dict_contains_all_required_keys(self):
        """Verify the output dictionary has the required keys."""
        res = FailureAnalysisResult(
            failure_type=FailureType.WRONG_TOOL,
            expected="search_knowledge_base",
            actual="web_search",
            evidence="Trace span #4",
            root_cause="[INFERRED] Agent selected an incorrect tool for a knowledge-base query.",
            recommendation="Review tool routing instructions.",
            confidence=0.92,
        )
        d = res.to_dict()
        assert d["failure_type"] == "Wrong tool selected"
        assert d["expected"] == "search_knowledge_base"
        assert d["actual"] == "web_search"
        assert d["evidence"] == "Trace span #4"
        assert "routing" in d["recommendation"]
        assert d["confidence"] == 0.92
        assert d["is_inferred"] is True

    def test_format_text_matches_spec(self):
        """Verify format_text prints the exact user-specified fields."""
        res = FailureAnalysisResult(
            failure_type=FailureType.WRONG_TOOL,
            expected="search_knowledge_base",
            actual="web_search",
            evidence="Trace span #4",
            root_cause="[INFERRED] Agent selected an incorrect tool for a knowledge-base query.",
            recommendation="Review tool routing instructions.",
            confidence=0.92,
        )
        formatted = res.format_text(include_disclaimer=False)
        expected_text = (
            "Failure:\nWrong tool selected\n\n"
            "Expected:\nsearch_knowledge_base\n\n"
            "Actual:\nweb_search\n\n"
            "Evidence:\nTrace span #4\n\n"
            "Likely Root Cause:\n[INFERRED] Agent selected an incorrect tool for a knowledge-base query.\n\n"
            "Recommendation:\nReview tool routing instructions."
        )
        assert formatted == expected_text

    def test_span_resolution_span_4(self):
        """Verify that when the 4th span is the culprit, it reports Trace span #4."""
        tc = TestCase(
            task_id="TC_004_SPAN",
            name="Multi Step Tool Selection",
            user_input="Look up internal policy",
            expected_behavior="Search policy in knowledge base",
            expected_tools=["search_knowledge_base"],
        )
        spans = [
            MockSpan(span_type="llm", tool_name=None),
            MockSpan(span_type="llm", tool_name=None),
            MockSpan(span_type="llm", tool_name=None),
            MockSpan(span_type="tool", tool_name="web_search"),  # 4th span (index 3) is culprit
        ]
        trace = Trace(
            task_id="TC_004_SPAN",
            query="Look up internal policy",
            final_answer="According to search...",
            spans=spans,
        )
        eval_results = [
            EvaluationResult(
                metric_name="tool_accuracy",
                score=0.0,
                passed=False,
                explanation="Expected tool search_knowledge_base, but web_search was called",
            )
        ]

        result = FailureAnalyzer.analyze(tc, trace, eval_results)
        assert result.failure_type == FailureType.WRONG_TOOL
        assert result.evidence == "Trace span #4"
        assert "web_search" in result.actual
        assert "search_knowledge_base" in result.expected

    def test_dict_based_telemetry_extraction(self):
        """Verify FailureAnalyzer can handle dict-based test cases, traces, and results."""
        tc_dict = {
            "task_id": "TC_DICT",
            "name": "Dict Test",
            "expected_tools": ["search_knowledge_base"],
            "expected_behavior": "Search KB",
        }
        trace_dict = {
            "task_id": "TC_DICT",
            "tools_called": ["web_search"],
            "steps": [
                {"step_index": 0, "tool_name": "web_search", "step_type": "tool"}
            ],
            "spans": [
                {"tool_name": "web_search", "span_type": "tool"}
            ],
        }
        eval_results_dict = [
            {
                "metric_name": "tool_accuracy",
                "score": 0.0,
                "passed": False,
                "explanation": "Expected tool search_knowledge_base, but web_search was called",
            }
        ]

        result = FailureAnalyzer.analyze(tc_dict, trace_dict, eval_results_dict)
        assert result.failure_type == FailureType.WRONG_TOOL
        assert "search_knowledge_base" in result.expected
        assert "web_search" in result.actual
        assert "Trace span #1" in result.evidence
        assert "[INFERRED]" in result.root_cause

