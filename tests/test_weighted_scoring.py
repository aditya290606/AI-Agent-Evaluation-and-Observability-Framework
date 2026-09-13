"""
Unit tests for the weighted scoring system.
"""

import pytest
from src.core.entities import EvaluationResult
from src.evaluation.scoring import (
    ScoringConfig,
    DEFAULT_WEIGHTS,
    calculate_case_scores,
)


def _make_eval_result(metric_name, score, passed, explanation=""):
    return EvaluationResult(
        metric_name=metric_name,
        score=score,
        passed=passed,
        threshold=0.7,
        explanation=explanation,
        evaluator_type="rule",
        evidence={},
    )


def test_calculate_case_scores_empty():
    res = calculate_case_scores([])
    assert res["passed"] is False
    assert res["weighted_score"] == 0.0
    assert len(res["failure_reasons"]) == 1
    assert "No evaluation metrics" in res["failure_reasons"][0]


def test_calculate_case_scores_balanced_all_pass():
    cfg = ScoringConfig.balanced_preset()
    results = [
        _make_eval_result("task_success", 1.0, True, "All good"),
        _make_eval_result("tool_selection_accuracy", 1.0, True, "Correct tools"),
        _make_eval_result("context_groundedness", 1.0, True, "Grounded"),
        _make_eval_result("semantic_answer_similarity", 0.9, True, "High similarity"),
        _make_eval_result("latency_budget", 1.0, True, "Under 2000ms"),
        _make_eval_result("cost_budget", 1.0, True, "Under $0.05"),
    ]
    summary = calculate_case_scores(results, cfg)
    assert summary["passed"] is True
    assert summary["weighted_score"] >= 95.0
    assert len(summary["failure_reasons"]) == 0
    assert summary["individual_scores"]["task_success"] == 1.0


def test_calculate_case_scores_failure_reasons():
    cfg = ScoringConfig.balanced_preset()
    results = [
        _make_eval_result("task_success", 1.0, True),
        _make_eval_result("tool_selection_accuracy", 0.0, False, "Wrong tool selected: web_search"),
        _make_eval_result("latency_budget", 0.0, False, "Latency 3500ms exceeded 2000ms budget"),
    ]
    summary = calculate_case_scores(results, cfg)
    # Expected failure reasons from the failed metrics
    assert len(summary["failure_reasons"]) == 2
    assert any("tool_selection_accuracy" in r and "Wrong tool" in r for r in summary["failure_reasons"])
    assert any("latency_budget" in r and "3500ms" in r for r in summary["failure_reasons"])


def test_critical_metric_failure_blocks_pass():
    # Even if weighted score is high, a critical metric failure (e.g. task_success) forces passed=False
    cfg = ScoringConfig(
        weights={"task_success": 0.1, "other_metric": 0.9},
        pass_threshold=0.5,
        critical_metrics=["task_success"],
    )
    results = [
        _make_eval_result("task_success", 0.0, False, "Task failed completely"),
        _make_eval_result("other_metric", 1.0, True, "Other passed"),
    ]
    summary = calculate_case_scores(results, cfg)
    # (0.0*0.1 + 1.0*0.9) / 1.0 = 0.90 -> 90% which is > 50%
    assert summary["weighted_score"] == 90.0
    # But because task_success is critical and failed, overall passed should be False!
    assert summary["passed"] is False
    assert len(summary["failure_reasons"]) == 1


def test_scoring_presets():
    quality_cfg = ScoringConfig.quality_first_preset()
    assert quality_cfg.pass_threshold == 0.75
    assert quality_cfg.weights["semantic_answer_similarity"] == 0.25

    eff_cfg = ScoringConfig.efficiency_preset()
    assert eff_cfg.weights["latency_budget"] == 0.25
    assert eff_cfg.weights["cost_budget"] == 0.20
