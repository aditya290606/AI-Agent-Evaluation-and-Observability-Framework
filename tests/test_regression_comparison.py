"""
Unit and integration tests for the Version & Experiment Comparison and Regression Detection system.
"""

import pytest
import pandas as pd
from src.analysis.regression import (
    VersionComparator,
    QualityGatePolicy,
    Direction,
    TransitionType,
)


def test_regression_detection_task_success_drop():
    """Verify regression detection on task success drop:
    Agent v1: Task Success: 94%
    Agent v2: Task Success: 86%
    Regression: -8 percentage points
    Status: FAILED QUALITY GATE
    """
    b_runs = [{"run_id": 1, "task_id": f"T{i}", "latency_ms": 100.0, "total_tokens": 500} for i in range(1, 101)]
    t_runs = [{"run_id": 2, "task_id": f"T{i}", "latency_ms": 100.0, "total_tokens": 500} for i in range(1, 101)]

    # v1: 94% pass rate
    b_evals = []
    for i in range(1, 101):
        passed = (i <= 94)
        b_evals.append({"run_id": 1, "task_id": f"T{i}", "metric_name": "task_success", "score": 1.0 if passed else 0.0, "passed": passed})

    # v2: 86% pass rate
    t_evals = []
    for i in range(1, 101):
        passed = (i <= 86)
        t_evals.append({"run_id": 2, "task_id": f"T{i}", "metric_name": "task_success", "score": 1.0 if passed else 0.0, "passed": passed})

    report = VersionComparator.compare(
        baseline_runs=b_runs,
        target_runs=t_runs,
        baseline_evals=b_evals,
        target_evals=t_evals,
        baseline_label="Agent v1",
        target_label="Agent v2",
    )

    ts_comp = report.metrics["task_success"]
    assert ts_comp.baseline_value == 94.0
    assert ts_comp.target_value == 86.0
    assert ts_comp.delta == -8.0
    assert ts_comp.is_regression is True
    assert report.quality_gate_passed is False
    assert report.quality_gate_status == "FAILED QUALITY GATE"
    assert any("Task Success" in r and "8.0" in r for r in report.quality_gate_reasons)


def test_performance_improvement_latency_reduction():
    """Verify detection of performance improvement:
    Example: Latency improved by 24%.
    """
    b_runs = [{"run_id": 1, "task_id": "T1", "latency_ms": 1000.0, "total_tokens": 500}]
    t_runs = [{"run_id": 2, "task_id": "T1", "latency_ms": 760.0, "total_tokens": 500}]
    evals = [{"run_id": 1, "task_id": "T1", "metric_name": "task_success", "score": 1.0, "passed": True},
             {"run_id": 2, "task_id": "T1", "metric_name": "task_success", "score": 1.0, "passed": True}]

    report = VersionComparator.compare(
        baseline_runs=b_runs,
        target_runs=t_runs,
        baseline_evals=evals,
        target_evals=evals,
    )

    lat_comp = report.metrics["latency"]
    assert lat_comp.baseline_value == 1000.0
    assert lat_comp.target_value == 760.0
    assert lat_comp.pct_change == -24.0
    assert lat_comp.is_improvement is True
    assert any("Latency improved by 24.0%" in h for h in report.improvement_highlights)
    assert report.quality_gate_passed is True
    assert report.quality_gate_status == "PASSED QUALITY GATE"


def test_critical_metric_regression_fails_quality_gate_even_if_overall_improves():
    """Critical requirement: Do not rely only on overall pass rate.
    A version can improve overall while regressing on a critical metric (e.g. tool accuracy).
    """
    b_runs = [{"run_id": 1, "task_id": "T1", "latency_ms": 500.0, "total_tokens": 500}]
    t_runs = [{"run_id": 2, "task_id": "T1", "latency_ms": 300.0, "total_tokens": 400}]

    # Baseline: overall score 75%, but tool accuracy was 100%
    b_evals = [
        {"run_id": 1, "task_id": "T1", "metric_name": "tool_selection_accuracy", "score": 1.0, "passed": True},
        {"run_id": 1, "task_id": "T1", "metric_name": "exact_answer", "score": 0.5, "passed": False},
    ]

    # Target: overall score improved to 80%, but tool accuracy regressed to 60%
    t_evals = [
        {"run_id": 2, "task_id": "T1", "metric_name": "tool_selection_accuracy", "score": 0.6, "passed": False},
        {"run_id": 2, "task_id": "T1", "metric_name": "exact_answer", "score": 1.0, "passed": True},
    ]

    report = VersionComparator.compare(
        baseline_runs=b_runs,
        target_runs=t_runs,
        baseline_evals=b_evals,
        target_evals=t_evals,
    )

    # Overall score improved (75% -> 80%)
    assert report.metrics["overall_score"].target_value > report.metrics["overall_score"].baseline_value
    # But Tool Accuracy regressed (100% -> 60%)
    assert report.metrics["tool_accuracy"].is_regression is True
    # Quality gate must fail!
    assert report.quality_gate_passed is False
    assert report.quality_gate_status == "FAILED QUALITY GATE"
    assert any("Tool Accuracy" in r for r in report.quality_gate_reasons)


def test_test_case_transition_diffing():
    """Verify test case categorization: newly failing, newly passing, persistent failures."""
    b_runs = [{"run_id": 1, "task_id": f"T{i}", "query": f"Query {i}"} for i in range(1, 5)]
    t_runs = [{"run_id": 2, "task_id": f"T{i}", "query": f"Query {i}"} for i in range(1, 5)]

    # T1: Passed in v1 -> Failed in v2 (NEWLY_FAILING)
    # T2: Failed in v1 -> Passed in v2 (NEWLY_PASSING)
    # T3: Failed in v1 -> Failed in v2 (PERSISTENT_FAIL)
    # T4: Passed in v1 -> Passed in v2 (PERSISTENT_PASS)
    b_evals = [
        {"run_id": 1, "task_id": "T1", "metric_name": "task_success", "score": 1.0, "passed": True},
        {"run_id": 1, "task_id": "T2", "metric_name": "task_success", "score": 0.0, "passed": False, "details": "Timeout in v1"},
        {"run_id": 1, "task_id": "T3", "metric_name": "task_success", "score": 0.0, "passed": False, "details": "Wrong answer"},
        {"run_id": 1, "task_id": "T4", "metric_name": "task_success", "score": 1.0, "passed": True},
    ]
    t_evals = [
        {"run_id": 2, "task_id": "T1", "metric_name": "task_success", "score": 0.0, "passed": False, "details": "Wrong tool in v2"},
        {"run_id": 2, "task_id": "T2", "metric_name": "task_success", "score": 1.0, "passed": True},
        {"run_id": 2, "task_id": "T3", "metric_name": "task_success", "score": 0.0, "passed": False, "details": "Still wrong answer"},
        {"run_id": 2, "task_id": "T4", "metric_name": "task_success", "score": 1.0, "passed": True},
    ]

    report = VersionComparator.compare(
        baseline_runs=b_runs,
        target_runs=t_runs,
        baseline_evals=b_evals,
        target_evals=t_evals,
    )

    assert len(report.newly_failing_tests) == 1
    assert report.newly_failing_tests[0].task_id == "T1"
    assert report.newly_failing_tests[0].transition_type == TransitionType.NEWLY_FAILING

    assert len(report.newly_passing_tests) == 1
    assert report.newly_passing_tests[0].task_id == "T2"
    assert report.newly_passing_tests[0].transition_type == TransitionType.NEWLY_PASSING

    assert len(report.persistent_failing_tests) == 1
    assert report.persistent_failing_tests[0].task_id == "T3"

    assert len(report.persistent_passing_tests) == 1
    assert report.persistent_passing_tests[0].task_id == "T4"


def test_all_9_metrics_comparison_populated():
    """Verify that all 9 required comparison metrics are calculated."""
    b_runs = [{"run_id": 1, "task_id": "T1", "latency_ms": 500.0, "total_tokens": 1000, "est_cost_usd": 0.002}]
    t_runs = [{"run_id": 2, "task_id": "T1", "latency_ms": 400.0, "total_tokens": 800, "est_cost_usd": 0.0016}]
    evals = [
        {"run_id": 1, "task_id": "T1", "metric_name": "task_success", "score": 1.0, "passed": True},
        {"run_id": 1, "task_id": "T1", "metric_name": "tool_selection_accuracy", "score": 1.0, "passed": True},
        {"run_id": 1, "task_id": "T1", "metric_name": "context_groundedness", "score": 0.9, "passed": True},
        {"run_id": 1, "task_id": "T1", "metric_name": "exact_answer", "score": 1.0, "passed": True},
        {"run_id": 2, "task_id": "T1", "metric_name": "task_success", "score": 1.0, "passed": True},
        {"run_id": 2, "task_id": "T1", "metric_name": "tool_selection_accuracy", "score": 1.0, "passed": True},
        {"run_id": 2, "task_id": "T1", "metric_name": "context_groundedness", "score": 0.95, "passed": True},
        {"run_id": 2, "task_id": "T1", "metric_name": "exact_answer", "score": 1.0, "passed": True},
    ]

    report = VersionComparator.compare(
        baseline_runs=b_runs,
        target_runs=t_runs,
        baseline_evals=evals,
        target_evals=evals,
    )

    required_keys = [
        "overall_score",
        "task_success",
        "tool_accuracy",
        "groundedness",
        "answer_quality",
        "latency",
        "token_usage",
        "cost",
        "error_rate",
    ]
    for k in required_keys:
        assert k in report.metrics, f"Missing metric {k}"
