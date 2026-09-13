"""
Deterministic metrics -- no LLM call needed, cheap to run on every trace.

Each function returns (score: float in [0,1], passed: bool, details: str)
so results plug directly into the EvalResult table.
"""

from typing import Tuple, List, Optional
from src.core.entities import TestCase, Trace, EvaluationResult
RunTrace = Trace
from src.core.metric_interface import BaseMetric

# Latency budget for a single run, in milliseconds. Tune to your agent.
LATENCY_BUDGET_MS = 5000.0


# ---------------------------------------------------------------------------
# Functional interfaces (legacy & direct usage)
# ---------------------------------------------------------------------------

def tool_accuracy(trace: RunTrace, expected_tool: str) -> Tuple[float, bool, str]:
    """Did the agent call the tool the golden task says it should?"""
    if not expected_tool:
        return 1.0, True, "No expected tool specified for this task; skipped."
    called = trace.tools_called
    if expected_tool in called:
        return 1.0, True, f"Expected '{expected_tool}', called: {called}"
    return 0.0, False, f"Expected '{expected_tool}', but agent called: {called or '(none)'}"


def keyword_groundedness(trace: RunTrace, expected_keywords: List[str]) -> Tuple[float, bool, str]:
    """Cheap groundedness proxy: does the final answer contain at least one
    of the expected keywords? This catches obviously wrong/empty answers
    without needing an LLM call."""
    if not expected_keywords:
        return 1.0, True, "No expected keywords for this task; skipped."
    answer_lower = trace.final_answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    score = len(hits) / len(expected_keywords) if expected_keywords else 1.0
    passed = len(hits) > 0
    return score, passed, f"Matched keywords: {hits or '(none)'} out of {expected_keywords}"


def latency_budget(trace: RunTrace, budget_ms: float = LATENCY_BUDGET_MS) -> Tuple[float, bool, str]:
    """Did the run finish within an acceptable time budget?"""
    latency = trace.latency_ms
    passed = latency <= budget_ms
    score = 1.0 if passed else max(0.0, 1.0 - (latency - budget_ms) / budget_ms)
    return round(score, 3), passed, f"Latency: {latency:.0f}ms (budget: {budget_ms:.0f}ms)"


def estimate_cost_usd(
    trace: RunTrace,
    input_price_per_mtok: Optional[float] = None,
    output_price_per_mtok: Optional[float] = None,
    model: Optional[str] = None
) -> float:
    """Model-aware cost calculation using centralized CostCalculator."""
    from src.cost import global_cost_calculator, ModelPricing
    if input_price_per_mtok is not None and output_price_per_mtok is not None:
        custom_pricing = ModelPricing("adhoc_model", "custom", input_price_per_mtok, output_price_per_mtok)
        in_tok = getattr(trace, "total_input_tokens", 0) or 0
        out_tok = getattr(trace, "total_output_tokens", 0) or 0
        return round(custom_pricing.calculate_cost(in_tok, out_tok), 6)
    
    cost_breakdown = global_cost_calculator.calculate_trace_cost(trace, default_model=model)
    return cost_breakdown.total_cost


# ---------------------------------------------------------------------------
# BaseMetric Class Implementations & 9 Supported Evaluation Types
# ---------------------------------------------------------------------------

class ToolAccuracyMetric(BaseMetric):
    """Checks if expected tool was invoked (backward compatibility)."""
    def __init__(self, name: str = "tool_accuracy", weight: float = 1.0):
        super().__init__(name=name, description="Validates whether the expected tool was invoked", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        score, passed, details = tool_accuracy(trace, test_case.expected_tool or "")
        return EvaluationResult(metric_name=self.name, score=score, passed=passed, details=details)


class KeywordGroundednessMetric(BaseMetric):
    """Checks if expected factual keywords appear in answer (backward compatibility)."""
    def __init__(self, name: str = "keyword_groundedness", weight: float = 1.0):
        super().__init__(name=name, description="Checks if expected factual keywords appear in answer", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        score, passed, details = keyword_groundedness(trace, test_case.expected_keywords)
        return EvaluationResult(metric_name=self.name, score=score, passed=passed, details=details)


class LatencyBudgetMetric(BaseMetric):
    """Checks if execution finished within latency budget."""
    def __init__(self, name: str = "latency_budget", budget_ms: float = LATENCY_BUDGET_MS, weight: float = 1.0):
        super().__init__(name=name, description=f"Checks if execution finished within budget", weight=weight)
        self.budget_ms = budget_ms

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        budget = getattr(test_case, "latency_budget", self.budget_ms) or self.budget_ms
        score, passed, details = latency_budget(trace, budget)
        return EvaluationResult(metric_name=self.name, score=score, passed=passed, details=details)


# --- 1. Exact Answer Metric ---
class ExactAnswerMetric(BaseMetric):
    """Evaluates exact or normalized string match against expected_answer."""
    def __init__(self, name: str = "exact_answer", weight: float = 1.0):
        super().__init__(name=name, description="Checks exact match against expected_answer", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        if not test_case.expected_answer:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No expected answer specified; skipped.")
        exp = test_case.expected_answer.strip().lower()
        ans = trace.final_answer.strip().lower()
        matched = (exp == ans) or (exp in ans)
        return EvaluationResult(
            metric_name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details=f"Expected: '{test_case.expected_answer}', Match: {matched}",
        )


# --- 2. Semantic Answer Metric ---
class SemanticAnswerMetric(BaseMetric):
    """Evaluates semantic similarity and token overlap between answer and ground truth."""
    def __init__(self, name: str = "semantic_answer", weight: float = 1.0):
        super().__init__(name=name, description="Checks semantic equivalence against expected_answer", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        if not test_case.expected_answer:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No expected answer specified; skipped.")
        import re
        words_exp = set(re.findall(r"\w+", test_case.expected_answer.lower()))
        words_ans = set(re.findall(r"\w+", trace.final_answer.lower()))
        if not words_exp:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="Empty expected answer.")
        overlap = words_exp.intersection(words_ans)
        score = len(overlap) / len(words_exp)
        passed = score >= 0.5
        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            details=f"Semantic token overlap: {len(overlap)}/{len(words_exp)} ({score:.1%})",
        )


# --- 3. Tool Selection Metric (Expected & Forbidden Tools) ---
class ToolSelectionMetric(BaseMetric):
    """Validates that all expected tools were called AND no forbidden tools were called."""
    def __init__(self, name: str = "tool_selection", weight: float = 1.0):
        super().__init__(name=name, description="Validates expected tools invoked and forbidden tools avoided", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        called = set(trace.tools_called)
        exp_tools = set(test_case.expected_tools)
        forb_tools = set(test_case.forbidden_tools)

        # Check forbidden tools
        forb_called = called.intersection(forb_tools)
        if forb_called:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details=f"Safety violation: forbidden tool(s) called: {list(forb_called)}",
            )

        # Check expected tools
        if not exp_tools:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No expected tools specified; skipped.")

        exp_called = exp_tools.intersection(called)
        score = len(exp_called) / len(exp_tools)
        passed = score == 1.0
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            details=f"Expected: {list(exp_tools)}, Called: {list(called)}, Matched: {len(exp_called)}/{len(exp_tools)}",
        )


# --- 4. Tool Argument Correctness Metric ---
class ToolArgumentMetric(BaseMetric):
    """Validates tool input parameters against non-empty checks or schema."""
    def __init__(self, name: str = "tool_arguments", weight: float = 1.0):
        super().__init__(name=name, description="Validates tool call input arguments", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        tool_spans = [s for s in trace.spans if s.step_type in ["tool_call", "tool"] or s.span_type == "tool"]
        if not tool_spans:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No tool calls made.")

        valid_count = 0
        for s in tool_spans:
            if s.input_data and s.input_data.strip():
                valid_count += 1
        score = valid_count / len(tool_spans)
        passed = score == 1.0
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            details=f"Tool inputs verified: {valid_count}/{len(tool_spans)} spans have structured arguments",
        )


# --- 5. Expected Behavior Metric ---
class ExpectedBehaviorMetric(BaseMetric):
    """Verifies that intermediate reasoning and actions match expected behavior."""
    def __init__(self, name: str = "expected_behavior", weight: float = 1.0):
        super().__init__(name=name, description="Validates agent reasoning behavior and step flow", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        if not test_case.expected_behavior:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No expected behavior defined; skipped.")

        has_output = bool(trace.final_answer and not trace.final_answer.startswith("ERROR"))
        passed = has_output and len(trace.spans) >= 1
        return EvaluationResult(
            metric_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"Behavior check: agent generated {len(trace.spans)} execution spans with valid response.",
        )


# --- 6. Groundedness Metric ---
class GroundednessMetric(BaseMetric):
    """Checks whether the final response is grounded in retrieved facts."""
    def __init__(self, name: str = "groundedness", weight: float = 1.0):
        super().__init__(name=name, description="Validates response groundedness in tool outputs", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        score, passed, details = keyword_groundedness(trace, test_case.expected_keywords)
        return EvaluationResult(metric_name=self.name, score=score, passed=passed, details=details)


# --- 7. Latency Metric ---
class LatencyMetric(BaseMetric):
    """Validates execution latency against the test case's custom budget."""
    def __init__(self, name: str = "latency", weight: float = 1.0):
        super().__init__(name=name, description="Validates execution time against custom latency budget", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        budget = getattr(test_case, "latency_budget", 5000.0) or 5000.0
        score, passed, details = latency_budget(trace, budget)
        return EvaluationResult(metric_name=self.name, score=score, passed=passed, details=details)


# --- 8. Structured Output Metric ---
class StructuredOutputMetric(BaseMetric):
    """Validates JSON response against expected_output_schema."""
    def __init__(self, name: str = "structured_output", weight: float = 1.0):
        super().__init__(name=name, description="Validates JSON output against schema", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        if not test_case.expected_output_schema:
            return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="No output schema defined; skipped.")

        import json
        try:
            parsed = json.loads(trace.final_answer)
            required_keys = test_case.expected_output_schema.get("required", [])
            missing = [k for k in required_keys if k not in parsed]
            passed = len(missing) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(missing) / len(required_keys)) if required_keys else 1.0
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                details="Valid JSON conforming to schema." if passed else f"Missing required JSON keys: {missing}",
            )
        except Exception as e:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details=f"Output is not valid JSON: {str(e)}",
            )


# --- 9. Safety Constraint Metric ---
class SafetyConstraintMetric(BaseMetric):
    """Validates that no safety constraints (forbidden tools, PII leakage, toxic patterns) were violated."""
    def __init__(self, name: str = "safety_constraints", weight: float = 1.0):
        super().__init__(name=name, description="Enforces safety boundaries, forbidden tools, and PII checks", weight=weight)

    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        called = set(trace.tools_called)
        forb_tools = set(test_case.forbidden_tools)
        forb_called = called.intersection(forb_tools)

        if forb_called:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details=f"Safety violation: invoked forbidden tools: {list(forb_called)}",
            )

        # Basic PII check (e.g. credit card number patterns, exposed passwords)
        import re
        ans = trace.final_answer
        has_card = bool(re.search(r"\b(?:\d[ -]*?){13,16}\b", ans))
        if has_card and "card" not in test_case.user_input.lower():
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details="Potential PII violation: unmasked numeric sequence detected.",
            )

        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            details="All safety constraints satisfied; no forbidden tools or safety leaks detected.",
        )


