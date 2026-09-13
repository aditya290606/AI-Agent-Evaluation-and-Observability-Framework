"""
OpenAI LLM Judge implementation with strict structured output parsing and graceful fallback.
"""

import os
import logging
from typing import Optional
from src.evaluation.judges.base import (
    LLMJudge,
    JudgeEvidenceContext,
    JudgeVerdict,
    parse_strict_judge_json,
)
from src.evaluation.judges.mock_judge import MockJudge

logger = logging.getLogger(__name__)

OPENAI_JUDGE_PROMPT = """You are an objective, rigorous AI evaluation judge checking an AI agent's final answer against factual evidence.

DO NOT blindly trust the agent's answer. Any claim, number, policy, or fact that is not directly verified by the evidence below is considered ungrounded or hallucinated.

EVIDENCE FOR THIS TASK:
User Query / Task Requirements:
"{query}"

Expected Answer (Ground Truth, if provided):
"{expected_answer}"

Expected Behavior (if provided):
"{expected_behavior}"

Retrieved Context Documents:
{retrieved_context}

Tool Execution Outputs:
{tool_outputs}

AGENT'S FINAL ANSWER:
"{final_answer}"

EVALUATION INSTRUCTIONS:
1. "correctness": Is the answer factually true according to the expected answer and evidence? (0.0 to 1.0)
2. "relevance": Does the answer directly answer the user's specific question? (0.0 to 1.0)
3. "groundedness": Is every substantive statement in the answer directly supported by the retrieved context or tool outputs? (0.0 to 1.0)
4. "completeness": Does the answer address all parts of the user request without omitting key requirements? (0.0 to 1.0)
5. "score": Overall composite score (0.0 to 1.0).
6. "passed": true if score >= {threshold}, otherwise false.
7. "reason": Concise 1-2 sentence explanation explaining the score and any gaps.
8. "evidence": A list of short string excerpts from the evidence supporting or contradicting the agent.

RESPOND STRICTLY WITH A JSON OBJECT MATCHING THIS SCHEMA. DO NOT INCLUDE ANY MARKDOWN FENCES OR EXTRA TEXT:
{{
  "score": <float 0.0-1.0>,
  "passed": <boolean>,
  "criteria": {{
    "correctness": <float 0.0-1.0>,
    "relevance": <float 0.0-1.0>,
    "groundedness": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>
  }},
  "reason": "<string>",
  "evidence": ["<string excerpt 1>", "<string excerpt 2>"]
}}
"""


class OpenAIJudge(LLMJudge):
    """
    Production-ready OpenAI LLM Judge with strict parsing and graceful mock fallback.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_to_mock: bool = True,
        name: str = "openai_judge",
    ):
        super().__init__(threshold=threshold, name=name)
        self.model = model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.fallback_to_mock = fallback_to_mock
        self._mock_judge = MockJudge(threshold=threshold)

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    def evaluate(self, context: JudgeEvidenceContext) -> JudgeVerdict:
        if not self.is_available:
            if self.fallback_to_mock:
                logger.warning("OpenAI API key not configured; falling back to MockJudge.")
                verdict = self._mock_judge.evaluate(context)
                verdict.reason = f"[MOCK FALLBACK: No OPENAI_API_KEY] {verdict.reason}"
                return verdict
            return JudgeVerdict(
                score=0.0,
                passed=False,
                criteria={"correctness": 0.0, "relevance": 0.0, "groundedness": 0.0, "completeness": 0.0},
                reason="OpenAIJudge unavailable: OPENAI_API_KEY is not set.",
                evidence=["Missing API key"],
                mode="mock",
                provider="openai",
                model=self.model,
                threshold=self.threshold,
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            retrieved_str = "\n---\n".join(context.retrieved_context) if context.retrieved_context else "(None)"
            tools_str = "\n".join(
                [f"- {t.get('tool_name')}: {str(t.get('output'))[:300]}" for t in context.tool_outputs]
            ) if context.tool_outputs else "(None)"

            prompt = OPENAI_JUDGE_PROMPT.format(
                query=context.query,
                expected_answer=context.expected_answer or "(None specified)",
                expected_behavior=context.expected_behavior or "(None specified)",
                retrieved_context=retrieved_str,
                tool_outputs=tools_str,
                final_answer=context.final_answer,
                threshold=self.threshold,
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict, objective AI evaluation judge."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=600,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = parse_strict_judge_json(raw_text, default_threshold=self.threshold)

            return JudgeVerdict(
                score=parsed["score"],
                passed=parsed["passed"],
                criteria=parsed["criteria"],
                reason=parsed["reason"],
                evidence=parsed["evidence"],
                mode="live",
                provider="openai",
                model=self.model,
                threshold=self.threshold,
            )

        except Exception as e:
            logger.error("OpenAIJudge call failed: %s", str(e), exc_info=True)
            if self.fallback_to_mock:
                verdict = self._mock_judge.evaluate(context)
                verdict.reason = f"[MOCK FALLBACK: OpenAI call failed ({type(e).__name__})] {verdict.reason}"
                return verdict

            return JudgeVerdict(
                score=0.0,
                passed=False,
                criteria={"correctness": 0.0, "relevance": 0.0, "groundedness": 0.0, "completeness": 0.0},
                reason=f"OpenAIJudge error: {str(e)}",
                evidence=[{"error": str(e)}],
                mode="mock",
                provider="openai",
                model=self.model,
                threshold=self.threshold,
            )
