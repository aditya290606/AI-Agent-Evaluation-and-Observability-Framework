"""
Base contracts and data structures for the pluggable LLM Judge architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import json
import re


@dataclass
class JudgeCriteria:
    """Detailed evaluation criteria scores (0.0 - 1.0)."""
    correctness: float = 0.0
    relevance: float = 0.0
    groundedness: float = 0.0
    completeness: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "correctness": round(float(self.correctness), 3),
            "relevance": round(float(self.relevance), 3),
            "groundedness": round(float(self.groundedness), 3),
            "completeness": round(float(self.completeness), 3),
        }


@dataclass
class JudgeVerdict:
    """Structured verdict returned by an LLM Judge."""
    score: float                          # 0.0 - 1.0 composite judge score
    passed: bool                          # True if score >= threshold
    criteria: Dict[str, float]            # Individual sub-criteria (correctness, relevance, groundedness, completeness)
    reason: str                           # Short, clear reasoning
    evidence: List[Any] = field(default_factory=list) # Cited factual evidence or excerpts
    mode: str = "mock"                    # "live" | "mock"
    provider: str = "mock"                # "mock" | "openai" | "anthropic"
    model: Optional[str] = None
    threshold: float = 0.70

    @property
    def evaluator_label(self) -> str:
        """Explicitly labels the judge as LIVE or MOCK."""
        return "LLM Judge — LIVE" if self.mode == "live" else "LLM Judge — MOCK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "passed": bool(self.passed),
            "criteria": self.criteria,
            "reason": str(self.reason),
            "evidence": self.evidence,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "evaluator_label": self.evaluator_label,
            "threshold": self.threshold,
        }


@dataclass
class JudgeEvidenceContext:
    """Context and evidence provided to the judge for evaluation."""
    query: str
    final_answer: str
    expected_answer: Optional[str] = None
    expected_behavior: Optional[str] = None
    retrieved_context: List[str] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    task_requirements: Optional[str] = None


class LLMJudge(ABC):
    """Abstract interface for all pluggable LLM Judge implementations."""

    def __init__(self, threshold: float = 0.70, name: str = "llm_judge"):
        self.threshold = threshold
        self.name = name

    @abstractmethod
    def evaluate(self, context: JudgeEvidenceContext) -> JudgeVerdict:
        """
        Evaluate the agent response against available evidence without blindly trusting it.
        Returns a structured JudgeVerdict.
        """
        pass

    def evaluate_trace(
        self,
        test_case: Any,
        trace: Any,
        execution_result: Optional[str] = None,
    ) -> JudgeVerdict:
        """
        Convenience adapter to extract evidence from a TestCase and Trace, then evaluate.
        """
        query = getattr(test_case, "user_input", None) or getattr(test_case, "query", "") or getattr(trace, "query", "")
        final_answer = execution_result or getattr(trace, "final_answer", "") or ""

        # Extract context and tool outputs from spans
        retrieved_context: List[str] = []
        tool_outputs: List[Dict[str, Any]] = []

        spans = getattr(trace, "spans", []) or getattr(trace, "steps", [])
        for s in spans:
            st = (getattr(s, "span_type", "") or getattr(s, "step_type", "")).lower()
            if st in ["retrieval"] and getattr(s, "output_data", ""):
                retrieved_context.append(str(s.output_data))
            elif st in ["tool", "tool_call"]:
                tool_outputs.append({
                    "tool_name": getattr(s, "tool_name", "unknown"),
                    "input": getattr(s, "input_data", ""),
                    "output": getattr(s, "output_data", ""),
                })

        context = JudgeEvidenceContext(
            query=str(query),
            final_answer=str(final_answer),
            expected_answer=getattr(test_case, "expected_answer", None),
            expected_behavior=getattr(test_case, "expected_behavior", None),
            retrieved_context=retrieved_context,
            tool_outputs=tool_outputs,
            task_requirements=getattr(test_case, "description", None),
        )
        return self.evaluate(context)


def parse_strict_judge_json(raw_text: str, default_threshold: float = 0.70) -> Dict[str, Any]:
    """
    Robust JSON parser for LLM judge outputs:
    Extracts JSON from raw text or markdown code blocks and validates required fields.
    """
    text = raw_text.strip()
    # Strip markdown code blocks if present
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Fallback: extract substring between first { and last }
    if "{" in text and "}" in text:
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        text = text[start_idx:end_idx]

    data = json.loads(text)

    # Validate & normalize fields
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    passed = bool(data.get("passed", score >= default_threshold))

    criteria_dict = data.get("criteria", {})
    if not isinstance(criteria_dict, dict):
        criteria_dict = {}

    criteria = {
        "correctness": max(0.0, min(1.0, float(criteria_dict.get("correctness", score)))),
        "relevance": max(0.0, min(1.0, float(criteria_dict.get("relevance", score)))),
        "groundedness": max(0.0, min(1.0, float(criteria_dict.get("groundedness", score)))),
        "completeness": max(0.0, min(1.0, float(criteria_dict.get("completeness", score)))),
    }

    reason = str(data.get("reason", data.get("reasoning", "Evaluation completed.")))
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []

    return {
        "score": score,
        "passed": passed,
        "criteria": criteria,
        "reason": reason,
        "evidence": evidence,
    }
