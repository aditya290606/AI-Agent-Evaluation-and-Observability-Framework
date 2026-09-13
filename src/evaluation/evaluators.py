"""
Modular Evaluator Architecture for AI Agents.

Implements the standard Evaluator interface:
    evaluate(test_case, execution_result, trace) -> EvaluationResult

Provides 14 separate evaluators across all key dimensions:
1. TaskSuccessEvaluator
2. ExactAnswerEvaluator
3. SemanticAnswerEvaluator
4. ToolSelectionEvaluator
5. ToolArgumentEvaluator
6. KeywordGroundednessEvaluator
7. ContextGroundednessEvaluator
8. LLMJudgeEvaluator
9. LatencyBudgetEvaluator
10. TokenUsageEvaluator
11. CostBudgetEvaluator
12. OutputSchemaEvaluator
13. ErrorRateEvaluator
14. RetryBehaviorEvaluator
"""

import json
import re
from typing import Optional, Any, List, Dict
from src.core.evaluator_interface import BaseEvaluator
from src.core.entities import TestCase, Trace, EvaluationResult
from src.evaluation import llm_judge as judge_module


def _normalize_text(text: str) -> str:
    """Lowercase and strip whitespace/punctuation for robust comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def _token_set(text: str) -> set:
    return set(_normalize_text(text).split())


# ===========================================================================
# 1. Task Success Evaluator
# ===========================================================================
class TaskSuccessEvaluator(BaseEvaluator):
    """Evaluates whether the agent accomplished the primary task without fatal errors."""

    def __init__(self, threshold: float = 1.0, weight: float = 0.30):
        super().__init__(
            name="task_success",
            threshold=threshold,
            weight=weight,
            evaluator_type="behavioral",
            description="Evaluates if agent successfully achieved task without fatal errors",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        answer = execution_result or (trace.final_answer if trace else "")
        if not answer or not str(answer).strip():
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation="Agent produced an empty response; task was not completed.",
                evidence={"answer_length": 0, "has_fatal_error": True},
                evaluator_type=self.evaluator_type,
            )

        # Check for error spans or exception text
        has_error = False
        error_msgs = []
        if trace:
            for s in trace.spans:
                if s.status == "error" or getattr(s, "error", None):
                    has_error = True
                    error_msgs.append(s.error or "Span failed")

        if has_error:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Execution encountered errors: {'; '.join(error_msgs[:2])}",
                evidence={"has_fatal_error": True, "errors": error_msgs},
                evaluator_type=self.evaluator_type,
            )

        # Check if required keywords match if specified
        if test_case and test_case.expected_keywords:
            ans_lower = str(answer).lower()
            matched = [kw for kw in test_case.expected_keywords if kw.lower() in ans_lower]
            if len(matched) == 0 and len(test_case.expected_keywords) > 0:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.5,
                    passed=False,
                    threshold=self.threshold,
                    explanation=f"Task executed without errors, but none of expected keywords were found.",
                    evidence={"expected_keywords": test_case.expected_keywords, "matched": []},
                    evaluator_type=self.evaluator_type,
                )

        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            threshold=self.threshold,
            explanation="Agent executed cleanly without errors and delivered an actionable answer.",
            evidence={"has_fatal_error": False, "answer_length": len(str(answer))},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 2. Exact Answer Evaluator
# ===========================================================================
class ExactAnswerEvaluator(BaseEvaluator):
    """Strict or normalized exact string match against expected_answer."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="exact_answer_correctness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Normalized exact match against expected answer",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not test_case or not test_case.expected_answer:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No ground-truth expected answer specified; skipped.",
                evidence={"expected": None, "actual": execution_result},
                evaluator_type=self.evaluator_type,
            )

        actual = str(execution_result or (trace.final_answer if trace else ""))
        expected = str(test_case.expected_answer)
        is_exact = _normalize_text(expected) in _normalize_text(actual)
        score = 1.0 if is_exact else 0.0
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            explanation="Exact answer matched ground-truth." if passed else "Exact answer mismatch.",
            evidence={"expected": expected, "actual": actual},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 3. Semantic Answer Evaluator
# ===========================================================================
class SemanticAnswerEvaluator(BaseEvaluator):
    """Evaluates semantic similarity and token overlap against expected_answer."""

    def __init__(self, threshold: float = 0.5, weight: float = 1.0):
        super().__init__(
            name="semantic_answer_similarity",
            threshold=threshold,
            weight=weight,
            evaluator_type="semantic",
            description="Semantic token similarity against expected answer",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not test_case or not test_case.expected_answer:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No ground truth expected answer specified; skipped.",
                evidence={"expected": None, "similarity": 1.0},
                evaluator_type=self.evaluator_type,
            )

        actual = str(execution_result or (trace.final_answer if trace else ""))
        expected = str(test_case.expected_answer)
        exp_tokens = _token_set(expected)
        act_tokens = _token_set(actual)

        if not exp_tokens:
            score = 1.0
        else:
            intersection = exp_tokens.intersection(act_tokens)
            score = len(intersection) / len(exp_tokens)

        passed = score >= self.threshold
        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Semantic token overlap: {score * 100:.1f}% (threshold: {self.threshold * 100:.0f}%)",
            evidence={"expected": expected, "actual": actual, "similarity": round(score, 3)},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 3b. Answer Correctness Evaluator (Unified Hackathon Metric)
# ===========================================================================
class AnswerCorrectnessEvaluator(BaseEvaluator):
    """Evaluates answer correctness combining exact ground-truth matching and heuristic token overlap.

    Returns the standard hackathon structure:
      {
        "metric": "Answer Correctness",
        "score": 0.0 to 1.0,
        "passed": bool,
        "threshold": float,
        "explanation": str,
        "evidence": dict | str
      }
    """

    def __init__(self, threshold: float = 0.80, weight: float = 0.20, name: str = "answer_correctness"):
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates answer correctness via exact ground-truth match and semantic token overlap",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        actual = str(execution_result or (trace.final_answer if trace else "") or "").strip()
        if not actual:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation="Agent produced an empty response; no answer to evaluate for correctness.",
                evidence={"expected": getattr(test_case, "expected_answer", None), "actual": ""},
                evaluator_type="deterministic",
            )

        # 1. Evaluate against expected_answer if provided
        if test_case and test_case.expected_answer:
            expected = str(test_case.expected_answer).strip()
            # Exact match check
            if _normalize_text(expected) in _normalize_text(actual):
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    explanation="Exact match: agent answer precisely matches expected ground-truth.",
                    evidence={"expected": expected, "actual": actual, "match_type": "exact"},
                    evaluator_type="deterministic",
                )

            # Heuristic token overlap check
            exp_tokens = _token_set(expected)
            act_tokens = _token_set(actual)
            if not exp_tokens:
                score = 1.0
            else:
                intersection = exp_tokens.intersection(act_tokens)
                recall = len(intersection) / len(exp_tokens)
                precision = len(intersection) / len(act_tokens) if act_tokens else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                score = round(max(recall, f1), 4)

            passed = score >= self.threshold
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                explanation=f"Heuristic answer overlap: {score * 100:.1f}% (threshold: {self.threshold * 100:.0f}%).",
                evidence={"expected": expected, "actual": actual, "overlap_score": score, "match_type": "heuristic"},
                evaluator_type="heuristic",
            )

        # 2. Evaluate against expected_keywords if provided
        if test_case and test_case.expected_keywords:
            keywords = test_case.expected_keywords
            act_lower = actual.lower()
            matched = [kw for kw in keywords if kw.lower() in act_lower]
            score = round(len(matched) / len(keywords), 4) if keywords else 1.0
            passed = score >= self.threshold
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                explanation=f"Keyword correctness: {len(matched)}/{len(keywords)} expected keywords identified in answer.",
                evidence={"expected_keywords": keywords, "matched": matched, "actual": actual, "match_type": "heuristic"},
                evaluator_type="heuristic",
            )

        # 3. No ground-truth specified: agent produced a substantive non-empty answer
        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            threshold=self.threshold,
            explanation="No ground-truth specified; agent produced a valid, substantive answer.",
            evidence={"actual": actual, "answer_length": len(actual)},
            evaluator_type="deterministic",
        )


# ===========================================================================
# 4. Tool Selection Evaluator
# ===========================================================================
class ToolSelectionEvaluator(BaseEvaluator):
    """Validates that expected tools were called and forbidden tools were avoided."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_selection_accuracy",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Validates tool selection matches requirements and avoids forbidden tools",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        tools_called = trace.tools_called if trace else []
        expected_tools = test_case.expected_tools if (test_case and test_case.expected_tools) else []
        forbidden_tools = test_case.forbidden_tools if (test_case and test_case.forbidden_tools) else []

        # Check forbidden tools first
        forbidden_called = [t for t in tools_called if t in forbidden_tools]
        if forbidden_called:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Forbidden tools were called: {forbidden_called}",
                evidence={
                    "expected": expected_tools,
                    "actual": tools_called,
                    "forbidden_called": forbidden_called,
                },
                evaluator_type=self.evaluator_type,
            )

        if not expected_tools:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No specific tools expected; zero forbidden tools called.",
                evidence={"expected": [], "actual": tools_called},
                evaluator_type=self.evaluator_type,
            )

        matched = [t for t in expected_tools if t in tools_called]
        score = len(matched) / len(expected_tools)
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Matched {len(matched)}/{len(expected_tools)} expected tools." if passed else f"Missing expected tools: {[t for t in expected_tools if t not in tools_called]}",
            evidence={"expected": expected_tools, "actual": tools_called, "matched": matched},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 4b. Tool Accuracy Evaluator (Unified Hackathon Metric)
# ===========================================================================
class ToolAccuracyEvaluator(BaseEvaluator):
    """Evaluates whether the agent selected the expected tool(s) and avoided forbidden tools.

    Returns the standard hackathon structure:
      {
        "metric": "Tool Accuracy",
        "score": 0.0 to 1.0,
        "passed": bool,
        "threshold": float,
        "explanation": str,
        "evidence": dict | str
      }
    """

    def __init__(self, threshold: float = 1.0, weight: float = 0.20, name: str = "tool_accuracy"):
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates tool selection accuracy against test expectations and avoids forbidden tools",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        tools_called = trace.tools_called if trace else []
        expected_tools = (
            test_case.expected_tools
            if (test_case and test_case.expected_tools)
            else ([test_case.expected_tool] if (test_case and getattr(test_case, "expected_tool", None)) else [])
        )
        forbidden_tools = test_case.forbidden_tools if (test_case and test_case.forbidden_tools) else []

        # Check forbidden tools first (fatal violation)
        forbidden_called = [t for t in tools_called if t in forbidden_tools]
        if forbidden_called:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Forbidden tools invoked: {forbidden_called}",
                evidence={
                    "expected": expected_tools,
                    "actual": tools_called,
                    "forbidden_called": forbidden_called,
                },
                evaluator_type="deterministic",
            )

        if not expected_tools:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No specific tools expected; zero forbidden tools called.",
                evidence={"expected": [], "actual": tools_called},
                evaluator_type="deterministic",
            )

        matched = [t for t in expected_tools if t in tools_called]
        score = len(matched) / len(expected_tools)
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Matched {len(matched)}/{len(expected_tools)} expected tools." if passed else f"Missing expected tools: {[t for t in expected_tools if t not in tools_called]}",
            evidence={"expected": expected_tools, "actual": tools_called, "matched": matched},
            evaluator_type="deterministic",
        )


# ===========================================================================
# 5. Tool Argument Evaluator
# ===========================================================================
class ToolArgumentEvaluator(BaseEvaluator):
    """Validates tool input parameters against non-empty checks or schema."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_argument_correctness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Validates tool call input arguments are non-empty and well-formed",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not trace:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No trace available; skipped.", evaluator_type=self.evaluator_type,
            )

        tool_spans = [s for s in trace.spans if (s.step_type in ["tool_call", "tool"] or getattr(s, "span_type", None) == "tool")]
        if not tool_spans:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No tool calls made in trace.",
                evidence={"tool_calls_count": 0},
                evaluator_type=self.evaluator_type,
            )

        valid_count = 0
        invalid_tools = []
        for s in tool_spans:
            val = str(s.input_data or "").strip()
            if val and val != "{}":
                valid_count += 1
            else:
                invalid_tools.append(s.tool_name or "unknown")

        score = valid_count / len(tool_spans)
        passed = score >= self.threshold
        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Valid arguments for {valid_count}/{len(tool_spans)} tool calls." if passed else f"Tool calls with empty/invalid arguments: {invalid_tools}",
            evidence={"valid_count": valid_count, "total": len(tool_spans), "invalid_tools": invalid_tools},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 6. Keyword Groundedness Evaluator
# ===========================================================================
class KeywordGroundednessEvaluator(BaseEvaluator):
    """Verifies that all expected keywords appear in the agent's answer."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="keyword_groundedness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Verifies required factual keywords exist in final answer",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not test_case or not test_case.expected_keywords:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No expected keywords specified.", evaluator_type=self.evaluator_type,
            )

        answer = str(execution_result or (trace.final_answer if trace else "")).lower()
        found = [kw for kw in test_case.expected_keywords if kw.lower() in answer]
        missing = [kw for kw in test_case.expected_keywords if kw.lower() not in answer]
        score = len(found) / len(test_case.expected_keywords)
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Found {len(found)}/{len(test_case.expected_keywords)} keywords." if passed else f"Missing required keywords: {missing}",
            evidence={"expected": test_case.expected_keywords, "found": found, "missing": missing},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 7. Context / Document Groundedness Evaluator
# ===========================================================================
class ContextGroundednessEvaluator(BaseEvaluator):
    """Evaluates whether answer content is grounded in tool execution outputs or retrieved context."""

    def __init__(self, threshold: float = 0.4, weight: float = 1.0):
        super().__init__(
            name="context_groundedness",
            threshold=threshold,
            weight=weight,
            evaluator_type="semantic",
            description="Evaluates whether answer is grounded in tool outputs and retrieved documents",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not trace:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No trace provided; skipped.", evaluator_type=self.evaluator_type,
            )

        # Collect all context from tool outputs and retrieval spans
        context_parts = []
        for s in trace.spans:
            if s.span_type in ["tool", "retrieval"] and s.output_data:
                context_parts.append(s.output_data)

        if not context_parts:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No tool or retrieval context emitted in trace.",
                evidence={"context_length": 0},
                evaluator_type=self.evaluator_type,
            )

        answer = str(execution_result or trace.final_answer or "")
        if not answer:
            for s in trace.spans:
                if s.span_type == "final_answer" and s.output_data:
                    answer = s.output_data
                    break

        combined_context = " ".join(context_parts)
        ctx_tokens = _token_set(combined_context)
        ans_tokens = _token_set(answer)

        if not ans_tokens:
            score = 0.0
        else:
            overlap = ans_tokens.intersection(ctx_tokens)
            score = min(1.0, len(overlap) / (len(ans_tokens) * 0.7))

        passed = score >= self.threshold
        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Context token grounding: {score * 100:.1f}% (threshold: {self.threshold * 100:.0f}%)",
            evidence={"context_chunks": len(context_parts), "grounding_score": round(score, 3)},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 7b. Groundedness Evaluator (Unified Hackathon Metric)
# ===========================================================================
class GroundednessEvaluator(BaseEvaluator):
    """Evaluates factual groundedness against context spans (tool outputs/retrieval) and key domain concepts.

    Returns the standard hackathon structure:
      {
        "metric": "Groundedness",
        "score": 0.0 to 1.0,
        "passed": bool,
        "threshold": float,
        "explanation": str,
        "evidence": dict | str
      }
    """

    def __init__(self, threshold: float = 0.70, weight: float = 0.20, name: str = "groundedness"):
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            evaluator_type="heuristic",
            description="Evaluates whether agent output is factually grounded in retrieval/tool context and expected domain concepts",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        actual = str(execution_result or (trace.final_answer if trace else "") or "").strip()
        if not actual:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation="Agent produced an empty response; cannot evaluate groundedness.",
                evidence={"groundedness_type": "none", "context_chunks": 0},
                evaluator_type="deterministic",
            )

        # 1. Extract context chunks from trace (retrieval, tool responses, mcp_tool outputs)
        context_parts: List[str] = []
        if trace and trace.spans:
            for s in trace.spans:
                if (
                    s.step_type in ["retrieval", "tool", "tool_call", "mcp_tool"]
                    or getattr(s, "span_type", None) in ["retrieval", "tool", "mcp_tool"]
                ):
                    out = str(s.output_data or "").strip()
                    if out and out != "{}" and len(out) > 2:
                        context_parts.append(out)

        # Case A: Context spans are available (RAG / Tool outputs)
        if context_parts:
            combined_context = " ".join(context_parts)
            ctx_tokens = _token_set(combined_context)
            ans_tokens = _token_set(actual)

            if not ans_tokens:
                ctx_score = 0.0
            else:
                overlap = ans_tokens.intersection(ctx_tokens)
                ctx_score = min(1.0, len(overlap) / max(1, len(ans_tokens) * 0.7))

            # Combine with keyword check if expected_keywords provided
            if test_case and test_case.expected_keywords:
                matched_kws = [kw for kw in test_case.expected_keywords if kw.lower() in actual.lower()]
                kw_score = len(matched_kws) / len(test_case.expected_keywords)
                score = round(0.6 * ctx_score + 0.4 * kw_score, 4)
            else:
                score = round(ctx_score, 4)

            passed = score >= self.threshold
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                explanation=f"Context grounding: {score * 100:.1f}% grounded in {len(context_parts)} context chunks (threshold: {self.threshold * 100:.0f}%).",
                evidence={"context_chunks": len(context_parts), "grounding_score": score, "grounding_type": "context_overlap"},
                evaluator_type="heuristic",
            )

        # Case B: No context spans, but expected_keywords specified
        if test_case and test_case.expected_keywords:
            matched_kws = [kw for kw in test_case.expected_keywords if kw.lower() in actual.lower()]
            score = round(len(matched_kws) / len(test_case.expected_keywords), 4) if test_case.expected_keywords else 1.0
            passed = score >= self.threshold
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                explanation=f"Keyword grounding: {len(matched_kws)}/{len(test_case.expected_keywords)} verified in response.",
                evidence={"expected_keywords": test_case.expected_keywords, "matched": matched_kws, "grounding_type": "keyword_verification"},
                evaluator_type="deterministic",
            )

        # Case C: Neither context nor keywords required
        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            threshold=self.threshold,
            explanation="No external context or keywords required; response is self-contained.",
            evidence={"context_chunks": 0, "grounding_type": "unconstrained"},
            evaluator_type="deterministic",
        )


# ===========================================================================
# 8. LLM-as-a-Judge Evaluator
# ===========================================================================
class LLMJudgeEvaluator(BaseEvaluator):
    """Model-graded evaluation assessing factual consistency, relevance, groundedness, and completeness."""

    def __init__(
        self,
        threshold: float = 0.70,
        weight: float = 1.0,
        provider: str = "auto",
        model: Optional[str] = None,
    ):
        super().__init__(
            name="llm_judge",
            threshold=threshold,
            weight=weight,
            evaluator_type="model_based",
            description="LLM-as-a-judge evaluation of answer factual consistency against available evidence",
        )
        self.provider = provider
        self.model = model

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not trace and not test_case:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No trace or test case provided; skipped.",
                evaluator_type="LLM Judge — MOCK",
                evidence={"criteria": {}, "evidence": []},
            )

        # Respect mock trace flag or configured provider
        use_mock = getattr(trace, "is_mock", True) if trace else True
        effective_provider = "mock" if use_mock and self.provider == "auto" else self.provider

        from src.evaluation.judges import get_llm_judge
        judge = get_llm_judge(
            provider=effective_provider,
            model=self.model,
            threshold=self.threshold,
        )

        verdict = judge.evaluate_trace(
            test_case=test_case,
            trace=trace,
            execution_result=str(execution_result) if execution_result is not None else None,
        )

        return EvaluationResult(
            metric_name=self.name,
            score=round(verdict.score, 4),
            passed=verdict.passed,
            threshold=self.threshold,
            explanation=verdict.reason,
            details=verdict.reason,
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


# ===========================================================================
# 9. Latency Evaluator (Unified Hackathon Metric)
# ===========================================================================
class LatencyEvaluator(BaseEvaluator):
    """Evaluates whether execution latency meets the test case latency budget.

    Returns the standard hackathon structure:
      {
        "metric": "Latency",
        "score": 0.0 to 1.0,
        "passed": bool,
        "threshold": float,
        "explanation": str,
        "evidence": dict | str
      }
    """

    def __init__(self, threshold: float = 1.0, weight: float = 0.10, name: str = "latency"):
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            evaluator_type="budget",
            description="Evaluates total execution latency against latency budget",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        budget = float(test_case.latency_budget if test_case and test_case.latency_budget else 5000.0)
        latency = float(trace.latency_ms if trace else 0.0)

        if latency <= budget:
            score = 1.0
            passed = True
            explanation = f"Execution completed in {latency:.1f}ms (budget: {budget:.0f}ms)."
        else:
            overage = latency - budget
            score = max(0.0, 1.0 - (overage / budget))
            passed = score >= self.threshold
            explanation = f"Exceeded latency budget by {overage:.1f}ms ({latency:.1f}ms vs {budget:.0f}ms)."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            evidence={"latency_ms": round(latency, 2), "budget_ms": budget, "overage_ms": max(0.0, round(latency - budget, 2))},
            evaluator_type="deterministic",
        )


class LatencyBudgetEvaluator(LatencyEvaluator):
    """Backward compatibility alias for LatencyEvaluator using 'latency_budget' as metric name."""

    def __init__(self, threshold: float = 1.0, weight: float = 0.10):
        super().__init__(threshold=threshold, weight=weight, name="latency_budget")


# ===========================================================================
# 10. Token Usage Evaluator
# ===========================================================================
class TokenUsageEvaluator(BaseEvaluator):
    """Assesses total tokens against maximum allowed token budget."""

    def __init__(self, token_budget: int = 4000, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="token_usage",
            threshold=threshold,
            weight=weight,
            evaluator_type="budget",
            description="Assesses token consumption against token budget",
        )
        self.token_budget = token_budget

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        in_tok = trace.total_input_tokens if trace else 0
        out_tok = trace.total_output_tokens if trace else 0
        total_tokens = in_tok + out_tok

        if total_tokens <= self.token_budget:
            score = 1.0
            passed = True
            explanation = f"Token usage within budget ({total_tokens} tokens <= {self.token_budget})."
        else:
            score = max(0.0, self.token_budget / total_tokens)
            passed = False
            explanation = f"Token budget exceeded: {total_tokens} tokens used (limit: {self.token_budget})."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            evidence={"total_tokens": total_tokens, "input_tokens": in_tok, "output_tokens": out_tok, "budget": self.token_budget},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 11. Cost Budget Evaluator
# ===========================================================================
class CostBudgetEvaluator(BaseEvaluator):
    """Evaluates estimated run cost against maximum cost budget."""

    def __init__(self, cost_budget_usd: float = 0.05, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="cost_budget",
            threshold=threshold,
            weight=weight,
            evaluator_type="budget",
            description="Evaluates estimated run cost against cost limit",
        )
        self.cost_budget_usd = cost_budget_usd

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        from src.cost import global_cost_calculator
        cost_breakdown = global_cost_calculator.calculate_trace_cost(trace)
        cost_usd = cost_breakdown.total_cost

        if cost_usd <= self.cost_budget_usd:
            score = 1.0
            passed = True
            explanation = f"Cost ${cost_usd:.5f} ({cost_breakdown.cost_type.lower()}) is within budget (${self.cost_budget_usd:.4f})."
        else:
            score = max(0.0, self.cost_budget_usd / cost_usd) if cost_usd > 0 else 0.0
            passed = False
            explanation = f"Cost budget exceeded: ${cost_usd:.5f} ({cost_breakdown.cost_type.lower()}) vs limit ${self.cost_budget_usd:.4f}."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            evidence={
                "estimated_cost_usd": round(cost_usd, 6),
                "budget_usd": self.cost_budget_usd,
                "cost_type": cost_breakdown.cost_type,
                "is_estimated": cost_breakdown.is_estimated,
                "model": cost_breakdown.model,
                "provider": cost_breakdown.provider,
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 12. Output Schema Evaluator
# ===========================================================================
class OutputSchemaEvaluator(BaseEvaluator):
    """Validates final answer JSON against test_case.expected_output_schema."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="output_schema_validation",
            threshold=threshold,
            weight=weight,
            evaluator_type="structural",
            description="Validates structured JSON output matches expected schema",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not test_case or not test_case.expected_output_schema:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No structured output schema required.", evaluator_type=self.evaluator_type,
            )

        text = str(execution_result or (trace.final_answer if trace else "")).strip()
        schema = test_case.expected_output_schema

        # Try to parse JSON from final answer
        try:
            # Handle possible markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            data = json.loads(text)
        except Exception as e:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Output is not valid JSON: {str(e)}",
                evidence={"error": str(e), "raw_text": text[:200]},
                evaluator_type=self.evaluator_type,
            )

        # Check required fields
        required = schema.get("required", [])
        missing = [f for f in required if f not in data]
        if missing:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Missing required schema fields: {missing}",
                evidence={"missing_fields": missing, "schema": schema},
                evaluator_type=self.evaluator_type,
            )

        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            threshold=self.threshold,
            explanation="Output conforms to required JSON schema.",
            evidence={"schema": schema},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 13. Error Rate Evaluator
# ===========================================================================
class ErrorRateEvaluator(BaseEvaluator):
    """Measures the proportion of execution spans that failed."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="error_rate",
            threshold=threshold,
            weight=weight,
            evaluator_type="reliability",
            description="Evaluates absence of execution errors across all spans",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not trace or not trace.spans:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No spans in trace; zero errors.", evaluator_type=self.evaluator_type,
            )

        error_spans = [s for s in trace.spans if s.status == "error" or getattr(s, "error", None)]
        total_spans = len(trace.spans)
        error_ratio = len(error_spans) / total_spans
        score = 1.0 - error_ratio
        passed = len(error_spans) == 0

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Zero errors across {total_spans} spans." if passed else f"{len(error_spans)}/{total_spans} spans encountered errors.",
            evidence={"error_count": len(error_spans), "total_spans": total_spans, "errors": [s.error for s in error_spans if s.error]},
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 14. Retry Behavior Evaluator
# ===========================================================================
class RetryBehaviorEvaluator(BaseEvaluator):
    """Detects repetitive looping or excessive tool call retries with identical arguments."""

    def __init__(self, max_allowed_retries: int = 2, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="retry_behavior",
            threshold=threshold,
            weight=weight,
            evaluator_type="behavioral",
            description="Detects whether agent gets stuck in excessive retry loops",
        )
        self.max_allowed_retries = max_allowed_retries

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        if not trace:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No trace provided; skipped.", evaluator_type=self.evaluator_type,
            )

        tool_spans = [s for s in trace.spans if (s.step_type in ["tool_call", "tool"] or getattr(s, "span_type", None) == "tool")]
        if len(tool_spans) <= 1:
            return EvaluationResult(
                metric_name=self.name, score=1.0, passed=True, threshold=self.threshold,
                explanation="No retry loops detected.", evaluator_type=self.evaluator_type,
            )

        # Count consecutive duplicate calls
        max_repeats = 1
        current_repeats = 1
        looping_tool = None

        for i in range(1, len(tool_spans)):
            prev = tool_spans[i - 1]
            curr = tool_spans[i]
            if prev.tool_name == curr.tool_name and str(prev.input_data).strip() == str(curr.input_data).strip():
                current_repeats += 1
                if current_repeats > max_repeats:
                    max_repeats = current_repeats
                    looping_tool = curr.tool_name
            else:
                current_repeats = 1

        if max_repeats > self.max_allowed_retries:
            score = max(0.0, 1.0 - (max_repeats - self.max_allowed_retries) * 0.3)
            passed = False
            explanation = f"Detected retry loop: tool '{looping_tool}' called {max_repeats} consecutive times with identical input."
        else:
            score = 1.0
            passed = True
            explanation = "Normal execution behavior; no excessive retry loops."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            evidence={"max_consecutive_repeats": max_repeats, "looping_tool": looping_tool},
            evaluator_type=self.evaluator_type,
        )


# Re-export specialized tool evaluators
from src.evaluation.tool_evaluators import (
    ToolSelectionAccuracyEvaluator,
    ToolArgumentCorrectnessEvaluator,
    ToolExecutionSuccessEvaluator,
    ToolEfficiencyEvaluator,
    UnnecessaryToolCallsEvaluator,
    ToolSequenceCorrectnessEvaluator,
    ToolRetryBehaviorEvaluator,
    ToolEvaluationSuiteEvaluator,
    extract_tool_telemetry,
)

# Re-export specialized MCP evaluators
from src.evaluation.mcp_evaluators import (
    McpToolSelectionEvaluator,
    McpArgumentCorrectnessEvaluator,
    McpToolSuccessEvaluator,
    McpUnnecessaryCallsEvaluator,
    McpLatencyEvaluator,
    McpFailureAnalysisEvaluator,
    McpToolSequenceEvaluator,
    McpEvaluationSuiteEvaluator,
    extract_mcp_telemetry,
    ALL_MCP_EVALUATORS,
)

# Backward-compatibility aliases
ToolSelectionEvaluator = ToolSelectionAccuracyEvaluator
ToolArgumentEvaluator = ToolArgumentCorrectnessEvaluator


# ===========================================================================
# Registry of standard evaluators
# ===========================================================================
EVALUATOR_REGISTRY = {
    # 5 Hackathon Core Metrics (Title Case & snake_case)
    "Task Success": TaskSuccessEvaluator,
    "task_success": TaskSuccessEvaluator,
    "Tool Accuracy": ToolAccuracyEvaluator,
    "tool_accuracy": ToolAccuracyEvaluator,
    "Answer Correctness": AnswerCorrectnessEvaluator,
    "answer_correctness": AnswerCorrectnessEvaluator,
    "Groundedness": GroundednessEvaluator,
    "groundedness": GroundednessEvaluator,
    "Latency": LatencyEvaluator,
    "latency": LatencyEvaluator,

    # Specific sub-metric evaluators & aliases
    "exact_answer": ExactAnswerEvaluator,
    "exact_answer_correctness": ExactAnswerEvaluator,
    "semantic_answer": SemanticAnswerEvaluator,
    "semantic_answer_similarity": SemanticAnswerEvaluator,
    "tool_selection": ToolAccuracyEvaluator,
    "tool_selection_accuracy": ToolAccuracyEvaluator,
    "tool_arguments": ToolArgumentCorrectnessEvaluator,
    "tool_argument_correctness": ToolArgumentCorrectnessEvaluator,
    "tool_execution_success": ToolExecutionSuccessEvaluator,
    "tool_success": ToolExecutionSuccessEvaluator,
    "tool_efficiency": ToolEfficiencyEvaluator,
    "unnecessary_tool_calls": UnnecessaryToolCallsEvaluator,
    "unnecessary_tools": UnnecessaryToolCallsEvaluator,
    "tool_sequence_correctness": ToolSequenceCorrectnessEvaluator,
    "tool_sequence": ToolSequenceCorrectnessEvaluator,
    "tool_retry_behavior": ToolRetryBehaviorEvaluator,
    "retry_behavior": ToolRetryBehaviorEvaluator,
    "tool_evaluation_suite": ToolEvaluationSuiteEvaluator,
    "tool_evaluation_overall": ToolEvaluationSuiteEvaluator,
    "mcp_tool_selection": McpToolSelectionEvaluator,
    "mcp_selection": McpToolSelectionEvaluator,
    "mcp_argument_correctness": McpArgumentCorrectnessEvaluator,
    "mcp_arguments": McpArgumentCorrectnessEvaluator,
    "mcp_tool_success": McpToolSuccessEvaluator,
    "mcp_success": McpToolSuccessEvaluator,
    "unnecessary_mcp_calls": McpUnnecessaryCallsEvaluator,
    "unnecessary_mcp": McpUnnecessaryCallsEvaluator,
    "mcp_latency": McpLatencyEvaluator,
    "mcp_failure_analysis": McpFailureAnalysisEvaluator,
    "mcp_failures": McpFailureAnalysisEvaluator,
    "mcp_tool_sequence": McpToolSequenceEvaluator,
    "mcp_sequence": McpToolSequenceEvaluator,
    "mcp_evaluation_suite": McpEvaluationSuiteEvaluator,
    "mcp_suite": McpEvaluationSuiteEvaluator,
    "keyword_groundedness": KeywordGroundednessEvaluator,
    "context_groundedness": ContextGroundednessEvaluator,
    "document_groundedness": ContextGroundednessEvaluator,
    "llm_judge": LLMJudgeEvaluator,
    "llm_judge_groundedness": LLMJudgeEvaluator,
    "latency_budget": LatencyBudgetEvaluator,
    "token_usage": TokenUsageEvaluator,
    "token_budget": TokenUsageEvaluator,
    "cost": CostBudgetEvaluator,
    "cost_budget": CostBudgetEvaluator,
    "output_schema": OutputSchemaEvaluator,
    "output_schema_validation": OutputSchemaEvaluator,
    "structured_output": OutputSchemaEvaluator,
    "error_rate": ErrorRateEvaluator,
}


def get_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """Factory helper to instantiate an evaluator by name with custom options."""
    cls = EVALUATOR_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown evaluator '{name}'. Available: {list(EVALUATOR_REGISTRY.keys())}")
    return cls(**kwargs)

