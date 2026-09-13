"""
Failure Analysis Module.

Produces a structured failure analysis for every failed evaluation,
classifying into one of 6 user-facing categories:

  1. Wrong Tool
  2. Incorrect Tool Arguments
  3. Incorrect Answer
  4. Poor Grounding
  5. Latency Violation
  6. Tool/API Failure

Each analysis clearly separates:
  - Observed evidence (deterministic telemetry)
  - Inferred diagnosis (heuristic hypothesis, never claimed as certain)

Usage:
    from src.analysis.failure_analysis import FailureAnalyzer, FailureAnalysisResult
    result = FailureAnalyzer.analyze(test_case, trace, eval_results)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureType(str, Enum):
    """The 6 user-facing failure categories."""
    WRONG_TOOL = "Wrong tool selected"
    INCORRECT_TOOL_ARGS = "Incorrect tool arguments"
    INCORRECT_ANSWER = "Incorrect answer"
    POOR_GROUNDING = "Poor grounding"
    LATENCY_VIOLATION = "Latency violation"
    TOOL_API_FAILURE = "Tool/API failure"
    NO_FAILURE = "No failure detected"


@dataclass
class FailureAnalysisResult:
    """Structured failure analysis for a single failed evaluation.

    IMPORTANT: `root_cause` and `recommendation` are inferred hypotheses,
    not deterministic facts. They must never be presented as certain.
    """
    failure_type: FailureType
    expected: str
    actual: str
    evidence: str                      # Specific span or metric reference (observed fact)
    root_cause: str                    # Inferred hypothesis — clearly labelled
    recommendation: str                # Actionable suggestion
    confidence: float = 0.0            # 0.0 – 1.0, how confident the heuristic is
    is_inferred: bool = True           # Always True for root_cause and recommendation
    additional_evidence: List[str] = field(default_factory=list)

    def format_text(self, include_disclaimer: bool = True) -> str:
        """Format the failure analysis matching the exact specification:

        Failure:
        <Failure Category>

        Expected:
        <Expected>

        Actual:
        <Actual>

        Evidence:
        <Evidence>

        Likely Root Cause:
        <Root Cause>

        Recommendation:
        <Recommendation>
        """
        lines = [
            f"Failure:\n{self.failure_type.value}",
            f"Expected:\n{self.expected}",
            f"Actual:\n{self.actual}",
            f"Evidence:\n{self.evidence}",
            f"Likely Root Cause:\n{self.root_cause}",
            f"Recommendation:\n{self.recommendation}",
        ]
        if include_disclaimer and self.is_inferred:
            lines.append(
                "Notice:\n[Observed Evidence is deterministic telemetry; Root Cause and Recommendation are inferred hypotheses, not certain facts.]"
            )
        return "\n\n".join(lines)

    def format_markdown(self) -> str:
        """Format as high-contrast markdown separating observed evidence from inferred diagnosis."""
        return (
            f"### 🔬 Failure Analysis: {self.failure_type.value}\n\n"
            f"**📋 Expected:**\n```\n{self.expected}\n```\n\n"
            f"**💥 Actual:**\n```\n{self.actual}\n```\n\n"
            f"**🔍 Evidence (Observed Telemetry):**\n`{self.evidence}`\n\n"
            f"**🧠 Likely Root Cause (Inferred Hypothesis):**\n> {self.root_cause}\n\n"
            f"**💡 Recommendation:**\n> {self.recommendation}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
            "root_cause": self.root_cause,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 2),
            "is_inferred": self.is_inferred,
            "additional_evidence": self.additional_evidence,
            "formatted_text": self.format_text(include_disclaimer=False),
        }


class FailureAnalyzer:
    """Deterministic heuristic failure classifier for failed evaluations.

    Analyzes trace spans and evaluation results to produce a structured
    FailureAnalysisResult. Uses only observed telemetry to determine
    failure type, then infers likely root cause (clearly labelled).
    """

    @classmethod
    def analyze(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        eval_results: Optional[List[Any]] = None,
    ) -> FailureAnalysisResult:
        """Produce a failure analysis from test case, trace, and eval results.

        Returns a FailureAnalysisResult with the most likely failure category.
        """
        # Collect observed facts from telemetry
        facts = cls._extract_facts(test_case, trace, eval_results)

        # Check if there's actually a failure
        has_failure = (
            bool(facts["failed_metrics"])
            or bool(facts["span_errors"])
            or facts["latency_exceeded"]
        )
        if not has_failure:
            return FailureAnalysisResult(
                failure_type=FailureType.NO_FAILURE,
                expected="All metrics pass",
                actual="All metrics passed",
                evidence="All trace spans completed successfully",
                root_cause="No failure detected",
                recommendation="No action required",
                confidence=1.0,
                is_inferred=False,
            )

        # Apply priority-ordered heuristic classification
        # Order: tool/API failure → wrong tool → incorrect args → poor grounding →
        #        latency violation → incorrect answer → fallback
        result = (
            cls._check_tool_api_failure(facts)
            or cls._check_wrong_tool(facts)
            or cls._check_incorrect_args(facts)
            or cls._check_poor_grounding(facts)
            or cls._check_latency_violation(facts)
            or cls._check_incorrect_answer(facts)
            or cls._fallback(facts)
        )
        return result

    @classmethod
    def analyze_all(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        eval_results: Optional[List[Any]] = None,
    ) -> List[FailureAnalysisResult]:
        """Produce ALL applicable failure analyses (not just the primary one)."""
        facts = cls._extract_facts(test_case, trace, eval_results)

        has_failure = (
            bool(facts["failed_metrics"])
            or bool(facts["span_errors"])
            or facts["latency_exceeded"]
        )
        if not has_failure:
            return []

        results = []
        for checker in [
            cls._check_tool_api_failure,
            cls._check_wrong_tool,
            cls._check_incorrect_args,
            cls._check_poor_grounding,
            cls._check_latency_violation,
            cls._check_incorrect_answer,
        ]:
            r = checker(facts)
            if r:
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Fact extraction (deterministic, zero speculation)
    # ------------------------------------------------------------------

    @classmethod
    def _extract_facts(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        eval_results: Optional[List[Any]],
    ) -> Dict[str, Any]:
        facts: Dict[str, Any] = {
            "expected_tools": [],
            "expected_answer": "",
            "expected_keywords": [],
            "expected_behavior": "",
            "latency_budget": 5000.0,
            "tools_called": [],
            "tool_args": [],
            "final_answer": "",
            "latency_ms": 0.0,
            "latency_exceeded": False,
            "span_errors": [],
            "failed_metrics": [],
            "failed_metric_details": {},
            "spans": [],
        }

        # From test case (handles objects and dicts)
        if test_case:
            if isinstance(test_case, dict):
                exp_tools = list(test_case.get("expected_tools", []) or [])
                if not exp_tools and test_case.get("expected_tool"):
                    exp_tools = [test_case["expected_tool"]]
                facts["expected_tools"] = exp_tools
                facts["expected_answer"] = test_case.get("expected_answer", "") or ""
                facts["expected_keywords"] = list(test_case.get("expected_keywords", []) or [])
                facts["expected_behavior"] = test_case.get("expected_behavior", "") or ""
                facts["latency_budget"] = float(test_case.get("latency_budget", 5000.0) or 5000.0)
                facts["expected_arguments"] = test_case.get("expected_tool_arguments") or test_case.get("expected_arguments")
            else:
                facts["expected_tools"] = list(getattr(test_case, "expected_tools", []) or [])
                if not facts["expected_tools"] and getattr(test_case, "expected_tool", None):
                    facts["expected_tools"] = [getattr(test_case, "expected_tool")]
                facts["expected_answer"] = getattr(test_case, "expected_answer", "") or ""
                facts["expected_keywords"] = list(getattr(test_case, "expected_keywords", []) or [])
                facts["expected_behavior"] = getattr(test_case, "expected_behavior", "") or ""
                facts["latency_budget"] = float(getattr(test_case, "latency_budget", 5000.0) or 5000.0)
                facts["expected_arguments"] = getattr(test_case, "expected_tool_arguments", None) or getattr(test_case, "expected_arguments", None)

        # From trace (handles objects and dicts)
        if trace:
            if isinstance(trace, dict):
                facts["final_answer"] = trace.get("final_answer", "") or ""
                facts["latency_ms"] = float(trace.get("latency_ms", 0.0) or 0.0)
                spans = trace.get("spans", []) or trace.get("steps", []) or []
                raw_tools = trace.get("tools_called", [])
            else:
                facts["final_answer"] = getattr(trace, "final_answer", "") or ""
                facts["latency_ms"] = float(getattr(trace, "latency_ms", 0.0) or 0.0)
                spans = getattr(trace, "spans", []) or getattr(trace, "steps", []) or []
                raw_tools = getattr(trace, "tools_called", [])

            if facts["latency_ms"] > facts["latency_budget"]:
                facts["latency_exceeded"] = True

            facts["spans"] = spans

            for idx, s in enumerate(spans):
                if isinstance(s, dict):
                    st = (s.get("span_type", "") or s.get("step_type", "")).lower()
                    tool_name = s.get("tool_name")
                    status = s.get("status", "success")
                    error = s.get("error")
                    inp = s.get("input_data") or s.get("input") or ""
                    outp = s.get("output_data") or s.get("output") or ""
                    lat = float(s.get("latency_ms", 0.0) or 0.0)
                else:
                    st = (getattr(s, "span_type", "") or getattr(s, "step_type", "")).lower()
                    tool_name = getattr(s, "tool_name", None)
                    status = getattr(s, "status", "success")
                    error = getattr(s, "error", None)
                    inp = getattr(s, "input_data", "") or getattr(s, "input", "") or ""
                    outp = getattr(s, "output_data", "") or getattr(s, "output", "") or ""
                    lat = float(getattr(s, "latency_ms", 0.0) or 0.0)

                # Collect tool calls
                if st in ("tool", "tool_call", "mcp_tool") or tool_name:
                    facts["tools_called"].append({
                        "index": idx + 1,
                        "span_ref": f"Trace span #{idx + 1}",
                        "tool_name": tool_name or "unknown",
                        "input": inp,
                        "output": outp,
                        "status": status,
                        "latency_ms": lat,
                    })
                    facts["tool_args"].append({
                        "tool_name": tool_name,
                        "arguments": inp,
                        "span_ref": f"Trace span #{idx + 1}",
                    })

                # Collect errors
                if error or status == "error":
                    facts["span_errors"].append({
                        "span_ref": f"Trace span #{idx + 1}",
                        "span_type": st,
                        "error": str(error or "Span status marked as error"),
                        "tool_name": tool_name,
                    })

            # Fallback if spans were empty but tools_called string or list exists
            if not facts["tools_called"] and raw_tools:
                if isinstance(raw_tools, str):
                    tool_list = [t.strip() for t in raw_tools.split(",") if t.strip()]
                else:
                    tool_list = list(raw_tools)
                for idx, t_name in enumerate(tool_list):
                    facts["tools_called"].append({
                        "index": idx + 1,
                        "span_ref": f"Trace span #{idx + 1}",
                        "tool_name": t_name,
                        "input": "",
                        "output": "",
                        "status": "success",
                        "latency_ms": 0.0,
                    })
                    facts["tool_args"].append({
                        "tool_name": t_name,
                        "arguments": "",
                        "span_ref": f"Trace span #{idx + 1}",
                    })

        # From evaluation results (handles objects and dicts)
        if eval_results:
            for er in eval_results:
                if isinstance(er, dict):
                    passed = er.get("passed", True)
                    m_name = er.get("metric_name") or er.get("metric") or "unknown"
                    explanation = er.get("explanation") or er.get("details") or ""
                    score = float(er.get("score", 0.0) or 0.0)
                    evidence = er.get("evidence", {}) or {}
                else:
                    passed = getattr(er, "passed", True)
                    m_name = getattr(er, "metric_name", "unknown")
                    explanation = getattr(er, "explanation", "") or getattr(er, "details", "") or ""
                    score = float(getattr(er, "score", 0.0) or 0.0)
                    evidence = getattr(er, "evidence", {}) or {}

                if not passed:
                    facts["failed_metrics"].append(m_name)
                    facts["failed_metric_details"][m_name] = {
                        "score": score,
                        "explanation": explanation,
                        "evidence": evidence,
                    }

        return facts

    # ------------------------------------------------------------------
    # Classification heuristics (each returns FailureAnalysisResult or None)
    # ------------------------------------------------------------------

    @classmethod
    def _check_tool_api_failure(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check for tool crashes, API errors, or infrastructure failures."""
        tool_errors = [e for e in facts["span_errors"] if e.get("span_type") in ("tool", "tool_call", "mcp_tool")]
        api_keywords = ["500", "502", "503", "connection", "timeout", "rate limit", "apierror"]
        api_errors = [
            e for e in facts["span_errors"]
            if any(k in e.get("error", "").lower() for k in api_keywords)
        ]

        errors = tool_errors or api_errors
        if not errors:
            return None

        err = errors[0]
        err_msg = err["error"][:200]
        is_api = bool(api_errors)

        return FailureAnalysisResult(
            failure_type=FailureType.TOOL_API_FAILURE,
            expected="Tool/API execution to complete with status 'success'",
            actual=f"{'API' if is_api else 'Tool'} raised error: {err_msg}",
            evidence=err["span_ref"],
            root_cause=(
                f"[INFERRED] The {'external API' if is_api else 'tool'} "
                f"'{err.get('tool_name', 'unknown')}' failed during execution. "
                f"This may indicate an infrastructure issue, invalid input, or upstream service outage."
            ),
            recommendation=(
                "Review the tool/API error logs. Verify input arguments are valid. "
                "Check upstream service health and implement retry logic with backoff."
            ),
            confidence=0.95,
            additional_evidence=[e["span_ref"] for e in errors[1:]],
        )

    @classmethod
    def _check_wrong_tool(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check if the agent selected the wrong tool vs expected."""
        expected = facts["expected_tools"]
        called = [t["tool_name"] for t in facts["tools_called"]]
        failed_metrics = facts["failed_metrics"]

        # Only flag if there ARE expected tools and none were called
        tool_metric_failed = any(
            m in ("tool_selection", "tool_accuracy", "tool_selection_accuracy", "Tool Accuracy")
            for m in failed_metrics
        )

        if expected and not any(t in called for t in expected):
            # Locate the specific wrong tool span
            wrong_calls = [t for t in facts["tools_called"] if t["tool_name"] not in expected]
            tool_span = wrong_calls[0]["span_ref"] if wrong_calls else (facts["tools_called"][0]["span_ref"] if facts["tools_called"] else "No tool span")
            actual_display = wrong_calls[0]["tool_name"] if len(wrong_calls) == 1 else (", ".join(called) if called else "No tools called")
            expected_display = expected[0] if len(expected) == 1 else ", ".join(expected)

            return FailureAnalysisResult(
                failure_type=FailureType.WRONG_TOOL,
                expected=expected_display,
                actual=actual_display,
                evidence=tool_span,
                root_cause=(
                    f"[INFERRED] Agent selected an incorrect tool ('{actual_display}') "
                    f"instead of expected '{expected_display}'. "
                    f"The tool routing logic may not have matched the query to the correct tool."
                ),
                recommendation=(
                    "Review tool routing instructions in the agent's system prompt. "
                    "Ensure tool descriptions clearly distinguish when each tool should be used."
                ),
                confidence=0.92,
            )
        elif tool_metric_failed and expected:
            # Tool was partially correct or sequence mismatch
            missing = [t for t in expected if t not in called]
            extra = [t for t in called if t not in expected]
            if missing or extra:
                detail = facts["failed_metric_details"].get(
                    next(m for m in failed_metrics if "tool" in m.lower()), {}
                )
                return FailureAnalysisResult(
                    failure_type=FailureType.WRONG_TOOL,
                    expected=", ".join(expected),
                    actual=", ".join(called),
                    evidence=detail.get("explanation", "Tool accuracy metric failed"),
                    root_cause=(
                        f"[INFERRED] Tool selection was partially incorrect. "
                        f"Missing: {missing or 'none'}. Extra: {extra or 'none'}. "
                        f"The agent may have called unnecessary tools or missed required ones."
                    ),
                    recommendation=(
                        "Review tool selection criteria. Consider adding explicit tool "
                        "routing examples in few-shot prompts."
                    ),
                    confidence=0.85,
                )

        return None

    @classmethod
    def _check_incorrect_args(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check if tool arguments were incorrect."""
        expected_args = facts.get("expected_arguments")
        if not expected_args:
            return None

        # Compare with actual tool arguments
        for tc in facts["tools_called"]:
            actual_input = tc.get("input", "")
            # Simple check: if expected arguments are not found in actual input
            mismatches = []
            if isinstance(expected_args, dict):
                for key, val in expected_args.items():
                    if str(val).lower() not in str(actual_input).lower():
                        mismatches.append(f"{key}={val}")

            if mismatches:
                return FailureAnalysisResult(
                    failure_type=FailureType.INCORRECT_TOOL_ARGS,
                    expected=f"Tool arguments: {expected_args}",
                    actual=f"Actual input: {actual_input[:200]}",
                    evidence=tc["span_ref"],
                    root_cause=(
                        f"[INFERRED] The agent passed incorrect arguments to "
                        f"'{tc['tool_name']}'. Missing/wrong values: {', '.join(mismatches)}. "
                        f"This may indicate incorrect query parsing or parameter extraction."
                    ),
                    recommendation=(
                        "Review how the agent extracts parameters from user queries. "
                        "Add input validation or structured output parsing for tool arguments."
                    ),
                    confidence=0.80,
                )

        return None

    @classmethod
    def _check_latency_violation(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check if execution exceeded the latency budget."""
        if not facts["latency_exceeded"]:
            # Also check if latency metric explicitly failed
            latency_failed = any(
                m in ("latency", "latency_budget", "Latency")
                for m in facts["failed_metrics"]
            )
            if not latency_failed:
                return None

        budget = facts["latency_budget"]
        actual = facts["latency_ms"]
        overshoot = actual - budget

        # Find the slowest span
        slowest_span = None
        for tc in facts["tools_called"]:
            if slowest_span is None or tc.get("latency_ms", 0) > slowest_span.get("latency_ms", 0):
                slowest_span = tc

        slow_ref = slowest_span["span_ref"] if slowest_span else "Overall trace"
        slow_name = slowest_span["tool_name"] if slowest_span else "unknown"

        return FailureAnalysisResult(
            failure_type=FailureType.LATENCY_VIOLATION,
            expected=f"Execution within {budget:.0f}ms budget",
            actual=f"Total latency: {actual:.0f}ms ({overshoot:+.0f}ms over budget)",
            evidence=slow_ref,
            root_cause=(
                f"[INFERRED] Execution exceeded latency budget by {overshoot:.0f}ms. "
                f"The slowest operation was '{slow_name}' at {slow_ref}. "
                f"This may indicate a slow external call, unoptimized tool, or excessive retries."
            ),
            recommendation=(
                "Profile the execution timeline to identify bottleneck spans. "
                "Consider caching, parallelization, or reducing max_tokens for LLM calls."
            ),
            confidence=0.94,
        )

    @classmethod
    def _check_incorrect_answer(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check if the final answer was incorrect."""
        answer_metrics = [
            m for m in facts["failed_metrics"]
            if m in (
                "answer_correctness", "exact_answer", "exact_answer_correctness",
                "semantic_answer_similarity", "semantic_answer", "Answer Correctness",
                "task_success", "Task Success",
            )
        ]
        if not answer_metrics:
            return None

        expected_answer = facts["expected_answer"]
        actual_answer = facts["final_answer"]

        # Get metric detail
        detail = facts["failed_metric_details"].get(answer_metrics[0], {})
        explanation = detail.get("explanation", "Answer did not match expected output")

        return FailureAnalysisResult(
            failure_type=FailureType.INCORRECT_ANSWER,
            expected=expected_answer[:300] if expected_answer else "Expected answer to match ground truth",
            actual=actual_answer[:300] if actual_answer else "No answer provided",
            evidence=explanation,
            root_cause=(
                f"[INFERRED] The agent's final answer did not match the expected output. "
                f"This may indicate incorrect reasoning, missing information from tool outputs, "
                f"or the agent summarizing/paraphrasing in an unexpected way."
            ),
            recommendation=(
                "Review the agent's reasoning chain. Check if tool outputs contained "
                "the correct information. Consider adding output format instructions."
            ),
            confidence=0.88,
        )

    @classmethod
    def _check_poor_grounding(cls, facts: Dict[str, Any]) -> Optional[FailureAnalysisResult]:
        """Check if the answer was poorly grounded in evidence."""
        grounding_metrics = [
            m for m in facts["failed_metrics"]
            if m in (
                "groundedness", "keyword_groundedness", "context_groundedness",
                "Groundedness",
            )
        ]
        if not grounding_metrics:
            return None

        detail = facts["failed_metric_details"].get(grounding_metrics[0], {})
        explanation = detail.get("explanation", "Answer not grounded in retrieved evidence")

        return FailureAnalysisResult(
            failure_type=FailureType.POOR_GROUNDING,
            expected="Answer grounded in tool outputs and retrieved context",
            actual="Answer contains claims not supported by execution evidence",
            evidence=explanation,
            root_cause=(
                f"[INFERRED] The agent's answer was not sufficiently grounded in "
                f"the retrieved context or tool outputs. The agent may have hallucinated "
                f"information or ignored relevant evidence from tool results."
            ),
            recommendation=(
                "Review retrieval quality. Ensure the agent's system prompt instructs "
                "it to base answers only on retrieved evidence. Add explicit grounding "
                "instructions."
            ),
            confidence=0.82,
        )

    @classmethod
    def _fallback(cls, facts: Dict[str, Any]) -> FailureAnalysisResult:
        """Fallback when no specific category matches."""
        failed = ", ".join(facts["failed_metrics"][:3]) or "unknown metrics"
        errors = facts["span_errors"]

        return FailureAnalysisResult(
            failure_type=FailureType.INCORRECT_ANSWER,
            expected="All evaluation metrics to pass",
            actual=f"Failed metrics: {failed}",
            evidence=errors[0]["span_ref"] if errors else "Evaluation results",
            root_cause=(
                f"[INFERRED] One or more evaluation metrics failed ({failed}). "
                f"The specific root cause could not be automatically determined "
                f"with high confidence. Manual inspection of the trace is recommended."
            ),
            recommendation=(
                "Inspect the full execution trace and evaluation breakdown. "
                "Check the Failures & RCA section for detailed AI-assisted diagnosis."
            ),
            confidence=0.50,
        )
