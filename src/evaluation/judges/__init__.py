"""
Pluggable LLM Judge package.
"""

import os
from typing import Optional
from src.evaluation.judges.base import (
    LLMJudge,
    JudgeCriteria,
    JudgeVerdict,
    JudgeEvidenceContext,
    parse_strict_judge_json,
)
from src.evaluation.judges.mock_judge import MockJudge
from src.evaluation.judges.openai_judge import OpenAIJudge
from src.evaluation.judges.anthropic_judge import AnthropicJudge


def get_llm_judge(
    provider: str = "auto",
    model: Optional[str] = None,
    threshold: float = 0.70,
    fallback_to_mock: bool = True,
    api_key: Optional[str] = None,
) -> LLMJudge:
    """
    Factory creating a pluggable LLM Judge.

    Providers:
    - "mock": Deterministic heuristic mock judge (always labeled LLM Judge — MOCK)
    - "openai": OpenAI LLM judge (gpt-4o-mini / custom model)
    - "anthropic": Anthropic Claude LLM judge (claude-3-5-haiku / custom model)
    - "auto": Auto-selects active API key, falling back gracefully to MockJudge
    """
    use_mock_env = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    prov = (provider or "auto").lower().strip()

    if prov == "mock":
        return MockJudge(threshold=threshold)

    if prov == "openai":
        return OpenAIJudge(
            threshold=threshold,
            model=model,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    if prov == "anthropic":
        return AnthropicJudge(
            threshold=threshold,
            model=model,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    # "auto" detection
    if use_mock_env:
        return MockJudge(threshold=threshold)

    if api_key or os.getenv("OPENAI_API_KEY"):
        return OpenAIJudge(
            threshold=threshold,
            model=model,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicJudge(
            threshold=threshold,
            model=model,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    return MockJudge(threshold=threshold)


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
]
