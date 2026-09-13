"""
Dedicated test suite for Hackathon Modular Evaluation Engine.

Verifies:
1. Five primary metrics:
   - Task Success (30%)
   - Tool Accuracy (20%)
   - Answer Correctness (20%)
   - Groundedness (20%)
   - Latency (10%)
2. Standard return dictionary structure:
   {
     "metric": "...",
     "score": 0.0,
     "passed": true,
     "threshold": 0.0,
     "explanation": "...",
     "evidence": "..."
   }
3. Weighted scoring calculation matching exact weights (30/20/20/20/10).
4. Four distinct evaluation taxonomies:
   - deterministic evaluation
   - heuristic evaluation
   - LLM-based evaluation
   - mock evaluation
5. Rule: Never present mock scores as real evaluation results.
"""

import pytest
from src.core.entities import TestCase, Trace, Span, EvaluationResult, EvaluationDataset
from src.core.agent_interface import BaseAgent
from src.core.engine import EvaluationEngine
from src.evaluation.evaluators import (
    TaskSuccessEvaluator,
    ToolAccuracyEvaluator,
    AnswerCorrectnessEvaluator,
    GroundednessEvaluator,
    LatencyEvaluator,
    EVALUATOR_REGISTRY,
    get_evaluator,
)
from src.evaluation.scoring import ScoringConfig, calculate_case_scores


def _make_trace(
    final_answer: str = "",
    latency_ms: float = 150.0,
    spans: list = None,
    is_mock: bool = False,
) -> Trace:
    t = Trace(task_id="TEST-HACKATHON", query="Sample test query", is_mock=is_mock)
    t.latency_ms = latency_ms
    t.final_answer = final_answer
    if spans:
        for s in spans:
            t.spans.append(s)
    return t


class DummyAgent(BaseAgent):
    def __init__(self, name="TestAgent", answer="The product costs 99 USD.", tools=None):
        super().__init__(name=name)
        self.answer = answer
        self.tools = tools or ["search_pricing"]

    def run(self, test_case: TestCase) -> Trace:
        spans = [
            Span(
                step_type="tool",
                tool_name=self.tools[0] if self.tools else "search_pricing",
                output_data="Pricing: Pro plan is 99 USD per month.",
                latency_ms=120.0,
            )
        ]
        return _make_trace(
            final_answer=self.answer,
            latency_ms=250.0,
            spans=spans,
            is_mock=False,
        )


def test_standard_return_schema_all_five_metrics():
    """Verify that every evaluator returns the required 6-key dictionary structure."""
    tc = TestCase(
        test_id="T_SCHEMA",
        user_input="How much is Pro plan?",
        expected_answer="99 USD",
        expected_tools=["search_pricing"],
        expected_keywords=["product", "99 USD"],
        latency_budget=1000.0,
    )
    trace = _make_trace(
        final_answer="The product costs 99 USD per month.",
        latency_ms=200.0,
        spans=[Span(step_type="tool", tool_name="search_pricing", output_data="Pro plan is 99 USD per month.")],
    )

    evaluators = [
        TaskSuccessEvaluator(),
        ToolAccuracyEvaluator(),
        AnswerCorrectnessEvaluator(),
        GroundednessEvaluator(),
        LatencyEvaluator(),
    ]

    expected_metrics = [
        "Task Success",
        "Tool Accuracy",
        "Answer Correctness",
        "Groundedness",
        "Latency",
    ]

    for ev, expected_metric_name in zip(evaluators, expected_metrics):
        res = ev(test_case=tc, trace=trace)

        # 1. Verify dictionary access protocol
        assert "metric" in res
        assert "score" in res
        assert "passed" in res
        assert "threshold" in res
        assert "explanation" in res
        assert "evidence" in res

        # 2. Verify exact metric title
        assert res["metric"] == expected_metric_name
        assert isinstance(res["score"], float)
        assert isinstance(res["passed"], bool)
        assert isinstance(res["threshold"], float)
        assert isinstance(res["explanation"], str) and len(res["explanation"]) > 0
        assert res["evidence"] is not None

        # 3. Verify to_dict output contains standard keys
        d = res.to_dict()
        assert d["metric"] == expected_metric_name
        assert d["score"] == res.score
        assert d["passed"] == res.passed
        assert d["threshold"] == res.threshold
        assert d["explanation"] == res.explanation


def test_hackathon_weights_and_overall_score():
    """Verify default weights: Task Success: 30%, Tool Accuracy: 20%, Answer Correctness: 20%, Groundedness: 20%, Latency: 10%."""
    cfg = ScoringConfig.hackathon_preset()
    assert cfg.weights["Task Success"] == 0.30
    assert cfg.weights["Tool Accuracy"] == 0.20
    assert cfg.weights["Answer Correctness"] == 0.20
    assert cfg.weights["Groundedness"] == 0.20
    assert cfg.weights["Latency"] == 0.10

    # Total weight must sum to exactly 1.0 (100%)
    total_w = sum([0.30, 0.20, 0.20, 0.20, 0.10])
    assert pytest.approx(total_w, 0.001) == 1.0

    # Test exact composite calculation:
    # Task Success: 1.0 (0.30)
    # Tool Accuracy: 1.0 (0.20)
    # Answer Correctness: 1.0 (0.20)
    # Groundedness: 0.5 (0.10)
    # Latency: 0.0 (0.00)
    # Expected weighted score = 0.30 + 0.20 + 0.20 + 0.10 + 0.0 = 0.80 => 80.0%
    results = [
        EvaluationResult(metric_name="Task Success", score=1.0, passed=True),
        EvaluationResult(metric_name="Tool Accuracy", score=1.0, passed=True),
        EvaluationResult(metric_name="Answer Correctness", score=1.0, passed=True),
        EvaluationResult(metric_name="Groundedness", score=0.5, passed=False),
        EvaluationResult(metric_name="Latency", score=0.0, passed=False),
    ]

    summary = calculate_case_scores(results, cfg)
    assert summary["weighted_score"] == 80.0
    assert summary["passed"] is True  # 80.0 >= 70.0% pass threshold and no critical failure
    assert len(summary["failure_reasons"]) == 2  # Groundedness and Latency failed


def test_task_success_evaluator_behaviors():
    """Test Task Success edge cases: empty answer, span errors, and missing keywords."""
    ev = TaskSuccessEvaluator()
    tc = TestCase(test_id="T_TS", user_input="Do task", expected_keywords=["confirmed"])

    # 1. Clean pass
    t_pass = _make_trace("Order confirmed successfully.")
    r_pass = ev(test_case=tc, trace=t_pass)
    assert r_pass["passed"] is True
    assert r_pass["score"] == 1.0
    assert r_pass["metric"] == "Task Success"

    # 2. Empty response
    t_empty = _make_trace("")
    r_empty = ev(test_case=tc, trace=t_empty)
    assert r_empty["passed"] is False
    assert r_empty["score"] == 0.0
    assert "empty" in r_empty["explanation"].lower()

    # 3. Span errors
    t_err = _make_trace(
        "Order confirmed",
        spans=[Span(step_type="tool", tool_name="order_db", status="error", error="Database connection refused")],
    )
    r_err = ev(test_case=tc, trace=t_err)
    assert r_err["passed"] is False
    assert r_err["score"] == 0.0
    assert "errors" in r_err["explanation"].lower()


def test_tool_accuracy_evaluator_behaviors():
    """Test Tool Accuracy: expected match, missing tool, and forbidden tool violations."""
    ev = ToolAccuracyEvaluator()
    tc = TestCase(
        test_id="T_TOOL",
        user_input="Fetch data",
        expected_tools=["database_query"],
        forbidden_tools=["drop_table"],
    )

    # 1. Expected tool called
    t_ok = _make_trace(spans=[Span(step_type="tool", tool_name="database_query")])
    r_ok = ev(test_case=tc, trace=t_ok)
    assert r_ok["passed"] is True
    assert r_ok["score"] == 1.0
    assert r_ok["metric"] == "Tool Accuracy"

    # 2. Forbidden tool invoked
    t_forbid = _make_trace(spans=[
        Span(step_type="tool", tool_name="database_query"),
        Span(step_type="tool", tool_name="drop_table"),
    ])
    r_forbid = ev(test_case=tc, trace=t_forbid)
    assert r_forbid["passed"] is False
    assert r_forbid["score"] == 0.0
    assert "forbidden" in r_forbid["explanation"].lower()

    # 3. Missing expected tool
    t_wrong = _make_trace(spans=[Span(step_type="tool", tool_name="web_search")])
    r_wrong = ev(test_case=tc, trace=t_wrong)
    assert r_wrong["passed"] is False
    assert r_wrong["score"] == 0.0
    assert "missing" in r_wrong["explanation"].lower()


def test_answer_correctness_deterministic_and_heuristic():
    """Test Answer Correctness distinguishes deterministic exact match and heuristic overlap."""
    ev = AnswerCorrectnessEvaluator(threshold=0.70)
    tc = TestCase(test_id="T_ANS", user_input="Price", expected_answer="The subscription fee is $50")

    # 1. Exact match -> deterministic
    t_exact = _make_trace("The subscription fee is $50 each month.")
    r_exact = ev(test_case=tc, trace=t_exact)
    assert r_exact["passed"] is True
    assert r_exact["score"] == 1.0
    assert r_exact["evaluation_type"] == "deterministic"

    # 2. Token overlap -> heuristic
    t_fuzzy = _make_trace("Subscription fee is approximately fifty dollars.")
    r_fuzzy = ev(test_case=tc, trace=t_fuzzy)
    assert r_fuzzy["evaluation_type"] == "heuristic"
    assert "heuristic" in r_fuzzy["explanation"].lower() or "overlap" in r_fuzzy["explanation"].lower()


def test_groundedness_evaluator_behaviors():
    """Test Groundedness against context retrieval chunks and keywords."""
    ev = GroundednessEvaluator(threshold=0.50)
    tc = TestCase(test_id="T_GROUND", user_input="Return policy")

    # Context chunks present in trace
    spans = [
        Span(step_type="retrieval", output_data="Refunds are accepted within 30 days of purchase."),
    ]
    t_grounded = _make_trace(
        final_answer="Customers can request refunds within 30 days of purchase.",
        spans=spans,
    )
    r_grounded = ev(test_case=tc, trace=t_grounded)
    assert r_grounded["passed"] is True
    assert r_grounded["score"] >= 0.50
    assert r_grounded["evaluation_type"] == "heuristic"
    assert r_grounded["metric"] == "Groundedness"


def test_latency_evaluator_behaviors():
    """Test Latency evaluator within budget vs over budget."""
    ev = LatencyEvaluator()
    tc = TestCase(test_id="T_LAT", user_input="Fast query", latency_budget=500.0)

    # Within budget
    t_fast = _make_trace(latency_ms=180.0)
    r_fast = ev(test_case=tc, trace=t_fast)
    assert r_fast["passed"] is True
    assert r_fast["score"] == 1.0
    assert r_fast["metric"] == "Latency"
    assert r_fast["evaluation_type"] == "deterministic"

    # Breached budget
    t_slow = _make_trace(latency_ms=1000.0)
    r_slow = ev(test_case=tc, trace=t_slow)
    assert r_slow["passed"] is False
    assert r_slow["score"] < 1.0
    assert "exceeded" in r_slow["explanation"].lower() or "breached" in r_slow["explanation"].lower()


def test_mock_evaluation_distinction_and_isolation():
    """CRITICAL RULE: Never present mock scores as real evaluation results."""
    ev = TaskSuccessEvaluator()
    tc = TestCase(test_id="T_MOCK", user_input="Simulated test")

    # Trace is marked as simulated / mock
    t_mock = _make_trace("Simulated answer", is_mock=True)
    r_mock = ev(test_case=tc, trace=t_mock)

    # 1. Result must explicitly declare itself as mock
    assert r_mock.is_mock is True
    assert r_mock.evaluation_type == "mock"
    assert r_mock["is_mock"] is True
    assert r_mock["evaluation_type"] == "mock"
    assert r_mock["explanation"].startswith("[MOCK EVALUATION]")

    # 2. Case scores must expose mock warning and mock type
    summary = calculate_case_scores([r_mock])
    assert summary["is_mock"] is True
    assert summary["evaluation_type"] == "mock"
    assert summary["mock_warning"] is not None
    assert "Mock scores must never be presented as real evaluation results" in summary["mock_warning"]


def test_evaluation_engine_integration_with_hackathon_metrics():
    """Verify that EvaluationEngine executes the 5 hackathon metrics by default."""
    agent = DummyAgent()
    tc = TestCase(
        test_id="HACK_1",
        user_input="What is the price?",
        expected_answer="99 USD",
        expected_tools=["search_pricing"],
        expected_keywords=["product"],
        latency_budget=1000.0,
    )
    ds = EvaluationDataset(name="hackathon_suite", test_cases=[tc])

    engine = EvaluationEngine(
        agent=agent,
        dataset=ds,
        persist=False,
    )

    # Verify default metrics are the 5 hackathon evaluators
    metric_names = [getattr(m, "name", "") for m in engine.metrics]
    assert "task_success" in metric_names
    assert "tool_accuracy" in metric_names
    assert "answer_correctness" in metric_names
    assert "groundedness" in metric_names
    assert "latency" in metric_names

    report = engine.run()
    assert report.total_test_cases == 1
    assert report.total_checks == 5
    assert report.overall_score > 0.0
    assert report.weighted_score > 0.0
    assert not report.is_mock
