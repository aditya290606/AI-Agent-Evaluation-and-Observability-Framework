"""
LLM-as-judge evaluator.

Pluggable LLM judge evaluating whether the agent's answer is factually
grounded in the tool outputs and retrieved context, without blindly trusting it.
Provides full backward-compatible access to legacy groundedness() and LLMJudgeGroundednessMetric.
"""

import os
import re
import json
from typing import Tuple

from src.core.entities import TestCase, Trace, EvaluationResult
from src.core.metric_interface import BaseMetric
from src.evaluation.judges import (
    LLMJudge,
    JudgeCriteria,
    JudgeVerdict,
    JudgeEvidenceContext,
    MockJudge,
    OpenAIJudge,
    AnthropicJudge,
    get_llm_judge,
    parse_strict_judge_json,
)

__all__ = [
    "LLMJudge",
    "JudgeCriteria",
    "JudgeVerdict",
    "JudgeEvidenceContext",
    "MockJudge",
    "OpenAIJudge",
    "AnthropicJudge",
    "get_llm_judge",
    "parse_strict_judge_json",
    "groundedness",
    "LLMJudgeGroundednessMetric",
]


def _mock_groundedness(query: str, tool_name: str, tool_output: str, final_answer: str):
    """Legacy backward-compatibility helper using MockJudge."""
    judge = MockJudge(threshold=0.70)
    ctx = JudgeEvidenceContext(
        query=query,
        final_answer=final_answer,
        tool_outputs=[{"tool_name": tool_name, "output": tool_output}],
    )
    verdict = judge.evaluate(ctx)
    return verdict.score, verdict.passed, verdict.reason


def _live_groundedness(query: str, tool_name: str, tool_output: str, final_answer: str):
    """Legacy backward-compatibility helper using live judge with fallback."""
    judge = get_llm_judge(provider="auto", threshold=0.70)
    ctx = JudgeEvidenceContext(
        query=query,
        final_answer=final_answer,
        tool_outputs=[{"tool_name": tool_name, "output": tool_output}],
    )
    verdict = judge.evaluate(ctx)
    return verdict.score, verdict.passed, verdict.reason


def groundedness(trace: Trace, use_mock: bool = True) -> Tuple[float, bool, str]:
    """
    Evaluates groundedness for a trace using the pluggable LLM judge.
    Returns (score, passed, reason).
    """
    provider = "mock" if use_mock else "auto"
    judge = get_llm_judge(provider=provider, threshold=0.70)
    verdict = judge.evaluate_trace(test_case=None, trace=trace)
    return verdict.score, verdict.passed, verdict.reason


class LLMJudgeGroundednessMetric(BaseMetric):
    """
    Production-ready semantic LLM-as-a-judge metric using pluggable judge architecture.
    """

    def __init__(
        self,
        name: str = "llm_judge_groundedness",
        use_mock: bool = True,
        weight: float = 1.0,
        provider: str = "auto",
        model: str = None,
    ):
        super().__init__(
            name=name,
            description="Evaluates whether final answer is factually grounded in tool output & context",
            weight=weight,
            threshold=0.70,
        )
        self.use_mock = use_mock
        prov = "mock" if use_mock else provider
        self.judge = get_llm_judge(provider=prov, model=model, threshold=self.threshold)
        self.evaluator_type = "LLM Judge — MOCK" if getattr(self.judge, "mode", "mock") == "mock" else "LLM Judge — LIVE"

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        verdict = self.judge.evaluate_trace(test_case=test_case, trace=trace)
        return EvaluationResult(
            metric_name=self.name,
            score=verdict.score,
            passed=verdict.passed,
            threshold=self.threshold,
            details=verdict.reason,
            explanation=verdict.reason,
            evidence={
                "criteria": verdict.criteria,
                "evidence": verdict.evidence,
                "provider": verdict.provider,
                "model": verdict.model,
                "mode": verdict.mode,
                "evaluator_label": verdict.evaluator_label,
            },
            evaluator_type=verdict.evaluator_label,
        )
