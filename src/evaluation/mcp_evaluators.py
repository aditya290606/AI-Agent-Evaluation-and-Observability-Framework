"""
MCP-Aware Evaluation Module for AI Agents.

Provides 7 specialized MCP evaluators + 1 composite suite:
1. McpToolSelectionEvaluator        - Evaluates whether the agent selected the expected MCP tool and server.
2. McpArgumentCorrectnessEvaluator  - Validates MCP tool input arguments against schema and expected parameters.
3. McpToolSuccessEvaluator          - Checks that all MCP tool executions succeeded without runtime/protocol errors.
4. McpUnnecessaryCallsEvaluator     - Detects superfluous or duplicate calls to MCP servers.
5. McpLatencyEvaluator              - Measures MCP roundtrip latency against performance budgets.
6. McpFailureAnalysisEvaluator      - Analyzes MCP error rates, resilience, and failure classifications.
7. McpToolSequenceEvaluator         - Evaluates the chronological sequence of MCP tool calls.
8. McpEvaluationSuiteEvaluator      - Composite suite calculating a weighted overall MCP score.
"""

import json
from difflib import SequenceMatcher
from typing import Optional, Any, List, Dict

from src.core.evaluator_interface import BaseEvaluator
from src.core.entities import (
    TestCase,
    Trace,
    Span,
    EvaluationResult,
    CapturedToolCall,
    ToolExecutionSummary,
)
from src.evaluation.tool_evaluators import extract_tool_telemetry, _safe_parse_args


def extract_mcp_telemetry(
    trace: Optional[Trace],
    test_case: Optional[TestCase] = None,
) -> List[CapturedToolCall]:
    """Extract only MCP-specific tool calls from the trace."""
    summary = extract_tool_telemetry(trace, test_case)
    return [c for c in summary.tool_calls if c.is_mcp]


# ===========================================================================
# 1. MCP Tool Selection Evaluator
# ===========================================================================
class McpToolSelectionEvaluator(BaseEvaluator):
    """Evaluates whether the agent invoked the expected MCP tool and server."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_tool_selection",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates whether the agent selected the correct MCP tool and server",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        actual_tools = [c.tool_name for c in mcp_calls]
        actual_servers = [c.mcp_server for c in mcp_calls if c.mcp_server]

        expected_tools = (
            test_case.expected_mcp_tools
            if (test_case and test_case.expected_mcp_tools)
            else (test_case.expected_tools if (test_case and test_case.expected_tools) else [])
        )
        expected_server = test_case.expected_mcp_server if test_case else None

        # If no MCP tools or servers expected
        if not expected_tools and not expected_server:
            if not mcp_calls:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    explanation="No MCP tools expected and zero MCP calls made.",
                    details="No MCP tools expected and zero MCP calls made.",
                    evidence={"expected_tools": [], "actual_tools": [], "mcp_calls": []},
                    evaluator_type=self.evaluator_type,
                )
            else:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    explanation=f"MCP tools called: {actual_tools} (no specific MCP tool required).",
                    details=f"MCP tools called: {actual_tools} (no specific MCP tool required).",
                    evidence={"expected_tools": [], "actual_tools": actual_tools, "mcp_calls": [c.to_dict() for c in mcp_calls]},
                    evaluator_type=self.evaluator_type,
                )

        if not mcp_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Expected MCP tool(s) {expected_tools} on server '{expected_server or 'any'}', but no MCP tools were called.",
                details=f"Expected MCP tool(s) {expected_tools} on server '{expected_server or 'any'}', but no MCP tools were called.",
                evidence={"expected_tools": expected_tools, "expected_server": expected_server, "actual_tools": []},
                evaluator_type=self.evaluator_type,
            )

        # Check server matching if expected
        server_matched = True
        if expected_server:
            server_matched = any(
                str(s).lower().strip() == str(expected_server).lower().strip()
                or str(expected_server).lower().strip() in str(s).lower().strip()
                for s in actual_servers
            )

        # Check tool matching
        matched_tools = [t for t in expected_tools if t in actual_tools]
        tool_score = len(matched_tools) / len(expected_tools) if expected_tools else 1.0

        if expected_server and not server_matched:
            final_score = round(tool_score * 0.5, 4)
            passed = False
            explanation = f"Wrong MCP server. Expected server '{expected_server}', but agent queried '{', '.join(set(actual_servers))}'."
        else:
            final_score = round(tool_score, 4)
            passed = final_score >= self.threshold
            if passed:
                explanation = f"Correct MCP tool(s) {matched_tools} selected on server '{expected_server or actual_servers[0]}'."
            else:
                missing = [t for t in expected_tools if t not in actual_tools]
                explanation = f"MCP tool selection mismatch. Missing expected MCP tool(s): {missing}. Actual calls: {actual_tools}."

        return EvaluationResult(
            metric_name=self.name,
            score=final_score,
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "expected_tools": expected_tools,
                "expected_server": expected_server,
                "actual_tools": actual_tools,
                "actual_servers": actual_servers,
                "matched_tools": matched_tools,
                "server_matched": server_matched,
                "mcp_calls": [c.to_dict() for c in mcp_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 2. MCP Argument Correctness Evaluator
# ===========================================================================
class McpArgumentCorrectnessEvaluator(BaseEvaluator):
    """Validates that MCP tool arguments are non-empty, JSON/schema-valid, and match expected arguments."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_argument_correctness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Validates MCP tool call arguments for validity, non-emptiness, and expected parameter values",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        if not mcp_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No MCP calls made in trace; zero argument defects.",
                details="No MCP calls made in trace; zero argument defects.",
                evidence={"mcp_calls_count": 0},
                evaluator_type=self.evaluator_type,
            )

        total_calls = len(mcp_calls)
        correct_count = 0
        defects = []

        for call in mcp_calls:
            args = call.arguments
            # Check 1: Non-empty validity
            is_valid = True
            if args is None:
                is_valid = False
            elif isinstance(args, dict):
                is_valid = len(args) > 0 or call.expected_arguments == {}
            elif isinstance(args, str):
                is_valid = bool(args.strip() and args.strip() != "{}")

            if not is_valid:
                defects.append(f"Position {call.sequence_position} ({call.tool_name}): Empty or invalid arguments.")
                continue

            # Check 2: Expected arguments if configured
            if call.expected_arguments:
                exp = call.expected_arguments
                if isinstance(exp, dict) and isinstance(args, dict):
                    mismatched = []
                    for k, v in exp.items():
                        if k not in args:
                            mismatched.append(f"Missing parameter '{k}'")
                        elif str(args[k]).lower().strip() != str(v).lower().strip():
                            if str(v).lower().strip() not in str(args[k]).lower().strip():
                                mismatched.append(f"Parameter '{k}' mismatch (expected '{v}', got '{args[k]}')")
                    if mismatched:
                        defects.append(f"Position {call.sequence_position} ({call.tool_name}): {'; '.join(mismatched)}")
                        continue
                elif str(exp).lower().strip() not in str(args).lower().strip():
                    defects.append(f"Position {call.sequence_position} ({call.tool_name}): Expected argument '{exp}' not found in actual '{args}'")
                    continue

            correct_count += 1

        score = correct_count / total_calls if total_calls > 0 else 1.0
        passed = score >= self.threshold

        if passed:
            explanation = f"All {total_calls} MCP tool call arguments were well-formed and valid."
        else:
            explanation = f"{total_calls - correct_count}/{total_calls} MCP calls had argument defects: {'; '.join(defects[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_mcp_calls": total_calls,
                "correct_mcp_calls": correct_count,
                "defects": defects,
                "mcp_calls": [c.to_dict() for c in mcp_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 3. MCP Tool Success Evaluator
# ===========================================================================
class McpToolSuccessEvaluator(BaseEvaluator):
    """Verifies that all MCP tool executions completed successfully without protocol or execution errors."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_tool_success",
            threshold=threshold,
            weight=weight,
            evaluator_type="reliability",
            description="Verifies that all MCP tool invocations completed successfully without errors",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        if not mcp_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No MCP calls executed; zero execution failures.",
                details="No MCP calls executed; zero execution failures.",
                evidence={"total_mcp_calls": 0, "failed_mcp_calls": 0},
                evaluator_type=self.evaluator_type,
            )

        failed_calls = [c for c in mcp_calls if c.execution_status != "success" or c.error]
        total_calls = len(mcp_calls)
        success_count = total_calls - len(failed_calls)

        score = success_count / total_calls if total_calls > 0 else 1.0
        passed = len(failed_calls) == 0

        if passed:
            explanation = f"All {total_calls} MCP tool invocations executed with 100% success rate."
        else:
            err_items = [f"{c.mcp_server or 'MCP'} -> {c.tool_name}: {c.error or 'failed'}" for c in failed_calls]
            explanation = f"{len(failed_calls)}/{total_calls} MCP tool calls failed: {'; '.join(err_items[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_mcp_calls": total_calls,
                "successful_mcp_calls": success_count,
                "failed_mcp_calls": len(failed_calls),
                "errors": [c.error for c in failed_calls if c.error],
                "mcp_calls": [c.to_dict() for c in mcp_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 4. Unnecessary MCP Calls Evaluator
# ===========================================================================
class McpUnnecessaryCallsEvaluator(BaseEvaluator):
    """Detects superfluous, duplicate, or unneeded MCP calls."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="unnecessary_mcp_calls",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Detects superfluous, redundant, or unneeded MCP calls",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        total_calls = len(mcp_calls)

        if total_calls == 0:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No MCP calls made; zero unnecessary MCP calls.",
                details="No MCP calls made; zero unnecessary MCP calls.",
                evidence={"unnecessary_count": 0, "total_calls": 0},
                evaluator_type=self.evaluator_type,
            )

        expected_tools = (
            test_case.expected_mcp_tools
            if (test_case and test_case.expected_mcp_tools)
            else (test_case.expected_tools if (test_case and test_case.expected_tools) else [])
        )

        unnecessary_count = 0
        unnecessary_details = []
        seen_calls = set()

        for idx, call in enumerate(mcp_calls):
            is_unnec = False
            reason = None

            # Duplicate call check
            arg_str = json.dumps(call.arguments, sort_keys=True) if isinstance(call.arguments, dict) else str(call.arguments)
            sig = f"{call.mcp_server}:{call.tool_name}:{arg_str}"
            if sig in seen_calls:
                is_unnec = True
                reason = f"Duplicate call #{idx+1} to MCP tool '{call.tool_name}' on server '{call.mcp_server}' with identical arguments."
            seen_calls.add(sig)

            # Unexpected tool check
            if expected_tools and call.tool_name not in expected_tools and not is_unnec:
                is_unnec = True
                reason = f"MCP tool '{call.tool_name}' was not in expected tools {expected_tools}."

            if is_unnec:
                unnecessary_count += 1
                unnecessary_details.append(reason)

        score = max(0.0, 1.0 - (unnecessary_count / total_calls)) if total_calls > 0 else 1.0
        passed = unnecessary_count == 0

        if passed:
            explanation = f"All {total_calls} MCP calls were necessary and relevant."
        else:
            explanation = f"⚠️ {unnecessary_count} potentially unnecessary MCP call{'s' if unnecessary_count != 1 else ''} detected: {'; '.join(unnecessary_details[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_mcp_calls": total_calls,
                "unnecessary_mcp_calls_count": unnecessary_count,
                "details": unnecessary_details,
                "mcp_calls": [c.to_dict() for c in mcp_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 5. MCP Latency Evaluator
# ===========================================================================
class McpLatencyEvaluator(BaseEvaluator):
    """Measures MCP roundtrip and execution latency against performance budgets."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_latency",
            threshold=threshold,
            weight=weight,
            evaluator_type="budget",
            description="Evaluates whether MCP server and tool execution latency complies with performance budget",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        budget = getattr(test_case, "mcp_latency_budget", 3000.0) if test_case else 3000.0

        if not mcp_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No MCP calls executed; 0 ms latency consumed.",
                details="No MCP calls executed; 0 ms latency consumed.",
                evidence={"total_mcp_latency_ms": 0.0, "budget_ms": budget},
                evaluator_type=self.evaluator_type,
            )

        total_lat = sum(c.latency for c in mcp_calls)
        max_lat = max(c.latency for c in mcp_calls)
        avg_lat = total_lat / len(mcp_calls)

        passed = total_lat <= budget
        if passed:
            score = 1.0
            explanation = f"MCP latency {total_lat:.1f} ms is within budget ({budget:.0f} ms). Avg: {avg_lat:.1f} ms, Max: {max_lat:.1f} ms."
        else:
            overage = total_lat - budget
            score = max(0.0, 1.0 - (overage / budget))
            explanation = f"MCP latency {total_lat:.1f} ms exceeded budget of {budget:.0f} ms by {overage:.1f} ms."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_mcp_latency_ms": round(total_lat, 2),
                "avg_mcp_latency_ms": round(avg_lat, 2),
                "max_mcp_latency_ms": round(max_lat, 2),
                "budget_ms": budget,
                "mcp_calls_count": len(mcp_calls),
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 6. MCP Failure Analysis Evaluator
# ===========================================================================
class McpFailureAnalysisEvaluator(BaseEvaluator):
    """Categorizes MCP failures and assesses agent resilience."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_failure_analysis",
            threshold=threshold,
            weight=weight,
            evaluator_type="reliability",
            description="Analyzes MCP failure patterns, categorizes error types, and verifies error handling",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        if not mcp_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No MCP calls executed; zero failure risk.",
                details="No MCP calls executed; zero failure risk.",
                evidence={"failure_count": 0, "categories": {}},
                evaluator_type=self.evaluator_type,
            )

        failed_calls = [c for c in mcp_calls if c.execution_status != "success" or c.error]
        categories: Dict[str, int] = {}

        for c in failed_calls:
            err_text = (c.error or "").lower()
            if "timeout" in err_text or "timed out" in err_text:
                cat = "Timeout / Latency SLA Breach"
            elif "connect" in err_text or "connection" in err_text or "server" in err_text:
                cat = "MCP Server Connection Error"
            elif "schema" in err_text or "argument" in err_text or "parameter" in err_text:
                cat = "MCP Tool Schema Validation Error"
            else:
                cat = "MCP Tool Execution Exception"
            categories[cat] = categories.get(cat, 0) + 1

        total_calls = len(mcp_calls)
        fail_count = len(failed_calls)
        score = (total_calls - fail_count) / total_calls if total_calls > 0 else 1.0
        passed = fail_count == 0

        if passed:
            explanation = "Zero MCP errors or failures detected across all server operations."
        else:
            cat_summary = ", ".join([f"{cat} ({cnt})" for cat, cnt in categories.items()])
            explanation = f"Detected {fail_count} MCP failure{'s' if fail_count != 1 else ''}: {cat_summary}."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_mcp_calls": total_calls,
                "failed_mcp_calls": fail_count,
                "failure_categories": categories,
                "failed_calls": [c.to_dict() for c in failed_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 7. MCP Tool Sequence Evaluator
# ===========================================================================
class McpToolSequenceEvaluator(BaseEvaluator):
    """Evaluates whether MCP server and tool operations occurred in the expected chronological sequence."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="mcp_tool_sequence",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates whether MCP tool calls occurred in the expected chronological order",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        mcp_calls = extract_mcp_telemetry(trace, test_case)
        actual_sequence = [c.tool_name for c in mcp_calls]

        expected_sequence = (
            test_case.expected_mcp_tool_sequence
            if (test_case and test_case.expected_mcp_tool_sequence)
            else (test_case.expected_mcp_tools if (test_case and test_case.expected_mcp_tools) else [])
        )

        if not expected_sequence:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No specific MCP sequence constraint defined.",
                details="No specific MCP sequence constraint defined.",
                evidence={"expected_sequence": [], "actual_sequence": actual_sequence},
                evaluator_type=self.evaluator_type,
            )

        if not actual_sequence:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Expected MCP tool sequence [{', '.join(expected_sequence)}], but no MCP tools were called.",
                details=f"Expected MCP tool sequence [{', '.join(expected_sequence)}], but no MCP tools were called.",
                evidence={"expected_sequence": expected_sequence, "actual_sequence": []},
                evaluator_type=self.evaluator_type,
            )

        matcher = SequenceMatcher(None, expected_sequence, actual_sequence)
        match_ratio = matcher.ratio()
        is_exact = actual_sequence == expected_sequence

        score = 1.0 if is_exact else match_ratio
        passed = score >= self.threshold

        if passed:
            explanation = f"MCP tool sequence matched expected order ({' ➔ '.join(actual_sequence)})."
        else:
            explanation = f"MCP sequence mismatch. Expected: [{' ➔ '.join(expected_sequence)}], Actual: [{' ➔ '.join(actual_sequence)}]."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "expected_sequence": expected_sequence,
                "actual_sequence": actual_sequence,
                "match_ratio": round(match_ratio, 3),
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 8. Composite MCP Evaluation Suite
# ===========================================================================
class McpEvaluationSuiteEvaluator(BaseEvaluator):
    """Runs all 7 specialized MCP evaluators and calculates an aggregated weighted composite score."""

    def __init__(self, threshold: float = 0.85, weight: float = 1.0):
        super().__init__(
            name="mcp_evaluation_suite",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Aggregated 7-dimension evaluation suite for MCP server and tool operations",
        )
        self.sub_evaluators = [
            McpToolSelectionEvaluator(),
            McpArgumentCorrectnessEvaluator(),
            McpToolSuccessEvaluator(),
            McpUnnecessaryCallsEvaluator(),
            McpLatencyEvaluator(),
            McpFailureAnalysisEvaluator(),
            McpToolSequenceEvaluator(),
        ]

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        sub_results: List[EvaluationResult] = []
        for ev in self.sub_evaluators:
            sub_results.append(ev.evaluate(test_case=test_case, execution_result=execution_result, trace=trace))

        weights = {
            "mcp_tool_selection": 0.25,
            "mcp_argument_correctness": 0.20,
            "mcp_tool_success": 0.20,
            "unnecessary_mcp_calls": 0.10,
            "mcp_latency": 0.10,
            "mcp_failure_analysis": 0.10,
            "mcp_tool_sequence": 0.05,
        }

        weighted_score = sum(res.score * weights.get(res.metric_name, 0.10) for res in sub_results)
        passed_all = all(res.passed for res in sub_results)
        passed = passed_all or (weighted_score >= self.threshold)

        failed_metrics = [res.metric_name for res in sub_results if not res.passed]
        if not failed_metrics:
            explanation = f"All 7 MCP evaluation dimensions passed with composite score {weighted_score * 100:.1f}%."
        else:
            explanation = f"Composite MCP score: {weighted_score * 100:.1f}%. Failed dimensions: {', '.join(failed_metrics)}."

        return EvaluationResult(
            metric_name=self.name,
            score=round(weighted_score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "sub_evaluations": {res.metric_name: res.to_dict() for res in sub_results},
                "failed_metrics": failed_metrics,
                "composite_score": round(weighted_score, 4),
            },
            evaluator_type=self.evaluator_type,
        )


ALL_MCP_EVALUATORS = [
    McpToolSelectionEvaluator,
    McpArgumentCorrectnessEvaluator,
    McpToolSuccessEvaluator,
    McpUnnecessaryCallsEvaluator,
    McpLatencyEvaluator,
    McpFailureAnalysisEvaluator,
    McpToolSequenceEvaluator,
    McpEvaluationSuiteEvaluator,
]
