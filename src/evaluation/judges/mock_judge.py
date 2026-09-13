"""
Mock LLM Judge implementation.

Deterministic, zero-cost heuristic judge checking response factual alignment
against tool outputs, retrieved context, expected answer, and task query.
"""

import re
from typing import Set, List, Dict, Any
from src.evaluation.judges.base import LLMJudge, JudgeEvidenceContext, JudgeVerdict


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


class MockJudge(LLMJudge):
    """
    Mock LLM Judge providing deterministic heuristic evaluations.
    Always explicitly marked with mode='mock' and evaluator_label='LLM Judge — MOCK'.
    """

    def __init__(self, threshold: float = 0.70, name: str = "mock_judge"):
        super().__init__(threshold=threshold, name=name)

    def evaluate(self, context: JudgeEvidenceContext) -> JudgeVerdict:
        answer = context.final_answer.strip()
        ans_tokens = _token_set(answer)
        evidence_citations: List[Any] = []

        if not answer:
            return JudgeVerdict(
                score=0.0,
                passed=False,
                criteria={"correctness": 0.0, "relevance": 0.0, "groundedness": 0.0, "completeness": 0.0},
                reason="Final answer was empty or missing.",
                evidence=["Empty agent response"],
                mode="mock",
                provider="mock",
                model="mock-heuristic",
                threshold=self.threshold,
            )

        # 1. Groundedness: overlap with tool outputs & retrieved context
        tool_texts = [str(t.get("output", "")) for t in context.tool_outputs if t.get("output")]
        combined_grounding = " ".join(context.retrieved_context + tool_texts)
        grounding_tokens = _token_set(combined_grounding)

        if grounding_tokens:
            overlap = ans_tokens.intersection(grounding_tokens)
            groundedness_score = min(1.0, len(overlap) / max(1, int(len(ans_tokens) * 0.5)))
            for ctx in context.retrieved_context[:2]:
                evidence_citations.append(f"Retrieved Context: {ctx[:120]}...")
            for t in context.tool_outputs[:2]:
                evidence_citations.append(f"Tool [{t.get('tool_name')}]: {str(t.get('output'))[:120]}...")
        else:
            # If no tools were called and no retrieval context exists
            groundedness_score = 1.0 if not context.expected_answer else 0.5
            evidence_citations.append("No tool output or retrieved documents provided.")

        # 2. Correctness: comparison against expected answer (if present)
        if context.expected_answer:
            exp_tokens = _token_set(context.expected_answer)
            correct_overlap = ans_tokens.intersection(exp_tokens)
            correctness_score = min(1.0, len(correct_overlap) / max(1, len(exp_tokens)))
            evidence_citations.append(f"Ground Truth Expected Answer: {context.expected_answer[:120]}")
        else:
            correctness_score = groundedness_score

        # 3. Relevance: overlap with user query / task requirements
        req_text = f"{context.query} {context.task_requirements or ''}"
        query_tokens = _token_set(req_text)
        rel_overlap = ans_tokens.intersection(query_tokens)
        relevance_score = min(1.0, len(rel_overlap) / max(1, min(len(query_tokens), 4)))

        # 4. Completeness: response depth and coverage of behavior
        completeness_score = min(1.0, len(ans_tokens) / 8.0)
        if context.expected_behavior:
            beh_tokens = _token_set(context.expected_behavior)
            if beh_tokens.intersection(ans_tokens):
                completeness_score = min(1.0, completeness_score + 0.2)

        # Composite score
        criteria = {
            "correctness": round(correctness_score, 3),
            "relevance": round(relevance_score, 3),
            "groundedness": round(groundedness_score, 3),
            "completeness": round(completeness_score, 3),
        }

        # Weighted composite: correctness (35%), groundedness (35%), relevance (20%), completeness (10%)
        composite_score = round(
            (correctness_score * 0.35)
            + (groundedness_score * 0.35)
            + (relevance_score * 0.20)
            + (completeness_score * 0.10),
            4,
        )

        passed = composite_score >= self.threshold
        reason = (
            f"(mock heuristic) Answer satisfies criteria with {composite_score * 100:.1f}% composite score."
            if passed
            else f"(mock heuristic) Answer failed threshold ({composite_score * 100:.1f}% < {self.threshold * 100:.0f}%)."
        )

        return JudgeVerdict(
            score=composite_score,
            passed=passed,
            criteria=criteria,
            reason=reason,
            evidence=evidence_citations,
            mode="mock",
            provider="mock",
            model="mock-heuristic",
            threshold=self.threshold,
        )
