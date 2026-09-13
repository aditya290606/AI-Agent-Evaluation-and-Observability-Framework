"""
Weighted scoring system for the evaluation engine.

Allows users to configure metric weights, calculate composite weighted scores,
derive pass/fail verdicts, and pinpoint detailed failure reasons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from src.core.entities import EvaluationResult


DEFAULT_WEIGHTS = {
    # 5 Hackathon Core Metrics
    "Task Success": 0.30,
    "task_success": 0.30,
    "Tool Accuracy": 0.20,
    "tool_accuracy": 0.20,
    "tool_selection_accuracy": 0.20,
    "tool_selection": 0.20,
    "Answer Correctness": 0.20,
    "answer_correctness": 0.20,
    "exact_answer_correctness": 0.20,
    "exact_answer": 0.20,
    "semantic_answer_similarity": 0.20,
    "semantic_answer": 0.20,
    "Groundedness": 0.20,
    "groundedness": 0.20,
    "keyword_groundedness": 0.20,
    "context_groundedness": 0.20,
    "Latency": 0.10,
    "latency": 0.10,
    "latency_budget": 0.10,
    # Additional extended metrics
    "cost_budget": 0.05,
    "cost": 0.05,
    "token_usage": 0.05,
    "output_schema_validation": 0.10,
    "error_rate": 0.15,
    "retry_behavior": 0.10,
}


@dataclass
class ScoringConfig:
    """Configuration for weighted scoring and pass/fail thresholds."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    pass_threshold: float = 0.70          # Minimum weighted score (0.0 to 1.0) required to pass
    critical_metrics: List[str] = field(default_factory=lambda: ["task_success", "Task Success", "safety_constraints"])

    @classmethod
    def hackathon_preset(cls) -> "ScoringConfig":
        """Hackathon 5-metric preset:
        Task Success: 30%, Tool Accuracy: 20%, Answer Correctness: 20%, Groundedness: 20%, Latency: 10%
        """
        return cls(
            weights={
                "Task Success": 0.30,
                "task_success": 0.30,
                "Tool Accuracy": 0.20,
                "tool_accuracy": 0.20,
                "tool_selection_accuracy": 0.20,
                "Answer Correctness": 0.20,
                "answer_correctness": 0.20,
                "exact_answer_correctness": 0.20,
                "semantic_answer_similarity": 0.20,
                "Groundedness": 0.20,
                "groundedness": 0.20,
                "keyword_groundedness": 0.20,
                "context_groundedness": 0.20,
                "Latency": 0.10,
                "latency": 0.10,
                "latency_budget": 0.10,
            },
            pass_threshold=0.70,
        )

    @classmethod
    def balanced_preset(cls) -> "ScoringConfig":
        """Default balanced preset matching hackathon specification."""
        return cls.hackathon_preset()

    @classmethod
    def quality_first_preset(cls) -> "ScoringConfig":
        return cls(
            weights={
                "task_success": 0.35,
                "context_groundedness": 0.25,
                "keyword_groundedness": 0.25,
                "semantic_answer_similarity": 0.25,
                "tool_selection_accuracy": 0.15,
            },
            pass_threshold=0.75,
        )

    @classmethod
    def efficiency_preset(cls) -> "ScoringConfig":
        return cls(
            weights={
                "task_success": 0.30,
                "latency_budget": 0.25,
                "cost_budget": 0.20,
                "token_usage": 0.10,
                "tool_selection_accuracy": 0.15,
            },
            pass_threshold=0.70,
        )


def calculate_case_scores(
    results: List[EvaluationResult],
    config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Computes individual scores, weighted composite score, overall unweighted score,
    pass/fail status, and structured failure reasons for a single test case run.
    """
    cfg = config or ScoringConfig()
    individual_scores: Dict[str, float] = {}
    failure_reasons: List[str] = []

    if not results:
        return {
            "individual_scores": {},
            "weighted_score": 0.0,
            "overall_score": 0.0,
            "passed": False,
            "failure_reasons": ["No evaluation metrics were executed."],
        }

    sum_weighted = 0.0
    total_weights = 0.0
    has_critical_failure = False

    for r in results:
        name = r.metric_name
        score = max(0.0, min(1.0, float(r.score)))
        individual_scores[name] = round(score, 4)

        weight = cfg.weights.get(name, 1.0)
        sum_weighted += score * weight
        total_weights += weight

        if not r.passed:
            expl = r.explanation or r.details or "Threshold not met"
            failure_reasons.append(f"[{name}] {expl}")
            if name in cfg.critical_metrics:
                has_critical_failure = True

    weighted_ratio = (sum_weighted / total_weights) if total_weights > 0 else (sum(individual_scores.values()) / len(individual_scores))
    overall_ratio = sum(individual_scores.values()) / len(individual_scores)

    # Pass condition: weighted score meets threshold and no critical failure
    passed = (weighted_ratio >= cfg.pass_threshold) and not has_critical_failure

    is_mock = any(getattr(r, "is_mock", False) or getattr(r, "evaluation_type", "") == "mock" for r in results)

    return {
        "individual_scores": individual_scores,
        "weighted_score": round(weighted_ratio * 100.0, 2),  # 0.0 - 100.0%
        "overall_score": round(overall_ratio * 100.0, 2),    # 0.0 - 100.0%
        "passed": passed,
        "pass_threshold_pct": round(cfg.pass_threshold * 100.0, 1),
        "failure_reasons": failure_reasons,
        "is_mock": is_mock,
        "evaluation_type": "mock" if is_mock else "real",
        "mock_warning": "SIMULATED RUN: Mock scores must never be presented as real evaluation results." if is_mock else None,
    }
