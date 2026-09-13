"""
Comprehensive Tool Evaluation Module for AI Agents.

Captures all 10 core tool telemetry fields:
1. tool_name
2. expected_tool
3. arguments
4. expected_arguments
5. execution_status
6. response
7. latency
8. retry_count
9. error
10. sequence_position

Provides 7 specialized evaluators + 1 composite suite:
1. ToolSelectionAccuracyEvaluator
2. ToolArgumentCorrectnessEvaluator
3. ToolExecutionSuccessEvaluator
4. ToolEfficiencyEvaluator
5. UnnecessaryToolCallsEvaluator
6. ToolSequenceCorrectnessEvaluator
7. ToolRetryBehaviorEvaluator
8. ToolEvaluationSuiteEvaluator
"""

import json
import re
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


def _safe_parse_args(input_data: Any) -> Any:
    """Safely parse input argument data to dict or structured object."""
    if input_data is None:
        return {}
    if isinstance(input_data, dict):
        return input_data
    if isinstance(input_data, (list, int, float, bool)):
        return input_data

    str_val = str(input_data).strip()
    if not str_val:
        return {}

    # Try JSON parsing
    try:
        if (str_val.startswith("{") and str_val.endswith("}")) or (
            str_val.startswith("[") and str_val.endswith("]")
        ):
            return json.loads(str_val)
    except Exception:
        pass

    return str_val


def extract_tool_telemetry(
    trace: Optional[Trace],
    test_case: Optional[TestCase] = None,
) -> ToolExecutionSummary:
    """Extract, parse, and structure every tool call in a trace into CapturedToolCall objects."""
    expected_tools = (
        test_case.expected_tools
        if test_case and test_case.expected_tools
        else ([test_case.expected_tool] if test_case and test_case.expected_tool else [])
    )
    expected_sequence = (
        test_case.expected_tool_sequence
        if test_case and test_case.expected_tool_sequence
        else list(expected_tools)
    )
    expected_args_map = (
        test_case.expected_tool_arguments
        if test_case and test_case.expected_tool_arguments
        else (test_case.expected_arguments if test_case and test_case.expected_arguments else None)
    )

    if not trace or not trace.spans:
        return ToolExecutionSummary(
            tool_calls=[],
            total_tool_calls=0,
            successful_tool_calls=0,
            failed_tool_calls=0,
            unnecessary_tool_calls_count=0,
            total_tool_latency_ms=0.0,
            actual_tool_sequence=[],
            expected_tool_sequence=list(expected_sequence),
            unnecessary_flags=[],
            retry_loops_detected=0,
            tools_called_names=[],
            is_selection_accurate=bool(not expected_tools),
            selection_reason="No tool calls recorded in trace." if expected_tools else "Zero tool calls expected and none made.",
        )

    # Filter tool spans (including MCP tool spans)
    tool_spans: List[Span] = []
    for s in trace.spans:
        is_tool = (
            s.step_type in ["tool_call", "tool", "mcp_tool", "mcp_call", "mcp"]
            or getattr(s, "span_type", None) in ["tool", "mcp_tool"]
            or bool(s.tool_name and s.step_type not in ["final_answer", "mcp_server"])
        )
        if is_tool and s.tool_name:
            tool_spans.append(s)

    captured_calls: List[CapturedToolCall] = []
    actual_sequence: List[str] = []
    tools_called_names: List[str] = []
    unnecessary_flags: List[str] = []

    retry_counts: Dict[str, int] = {}
    last_call_signature: Optional[str] = None
    current_consecutive_repeats = 1
    max_consecutive_repeats = 1

    # Map expected tools by sequence index if available
    for idx, span in enumerate(tool_spans):
        tool_name = span.tool_name or "unknown_tool"
        actual_sequence.append(tool_name)
        tools_called_names.append(tool_name)

        # 1. Expected tool for this sequence position
        exp_tool = None
        if idx < len(expected_sequence):
            exp_tool = expected_sequence[idx]
        elif expected_tools:
            # Fallback to general expected tools if single or unordered
            exp_tool = expected_tools[0] if len(expected_tools) == 1 else None

        # 2. Parse arguments & expected arguments
        parsed_args = _safe_parse_args(span.input_data)
        exp_args = None
        if isinstance(expected_args_map, dict):
            if tool_name in expected_args_map:
                exp_args = expected_args_map[tool_name]
            elif idx < len(expected_sequence) and expected_sequence[idx] in expected_args_map:
                exp_args = expected_args_map[expected_sequence[idx]]
            elif "arguments" in expected_args_map:
                exp_args = expected_args_map["arguments"]
            elif idx == 0 and len(expected_tools) <= 1:
                exp_args = expected_args_map

        # 3. Execution status and error
        status = (span.status or "success").lower()
        err_msg = span.error
        if status in ["error", "failed"] and not err_msg:
            err_msg = span.output_data if "error" in str(span.output_data).lower() else "Tool execution failed"
        elif err_msg and status == "success":
            status = "error"

        # 4. Latency
        lat = span.latency_ms if span.latency_ms > 0 else (span.duration_ms or 0.0)

        # 5. Retry calculation
        arg_str = json.dumps(parsed_args, sort_keys=True) if isinstance(parsed_args, dict) else str(parsed_args)
        call_sig = f"{tool_name}:{arg_str}"

        if call_sig == last_call_signature:
            current_consecutive_repeats += 1
            if current_consecutive_repeats > max_consecutive_repeats:
                max_consecutive_repeats = current_consecutive_repeats
            retry_count = current_consecutive_repeats - 1
        else:
            current_consecutive_repeats = 1
            retry_count = retry_counts.get(call_sig, 0)
            retry_counts[call_sig] = retry_count + 1

        last_call_signature = call_sig

        # 6. Check if this call is potentially unnecessary
        is_unnec = False
        unnec_reason = None

        if expected_tools:
            if tool_name not in expected_tools:
                is_unnec = True
                unnec_reason = f"Tool '{tool_name}' was not in expected tools {expected_tools}."
            elif actual_sequence.count(tool_name) > expected_sequence.count(tool_name) and expected_sequence.count(tool_name) > 0:
                is_unnec = True
                unnec_reason = f"Tool '{tool_name}' called {actual_sequence.count(tool_name)} times, but expected only {expected_sequence.count(tool_name)}."
        elif test_case and not expected_tools and not expected_sequence:
            # When zero tools are expected, any tool call is unnecessary
            is_unnec = True
            unnec_reason = f"No tool calls were expected for this task, but '{tool_name}' was invoked."

        # Redundant consecutive duplicate retry
        if retry_count > 1 and not is_unnec:
            is_unnec = True
            unnec_reason = f"Duplicate repetitive call #{retry_count + 1} with identical arguments."

        if is_unnec:
            unnecessary_flags.append(f"Position {idx + 1} ({tool_name}): {unnec_reason}")

        # MCP context
        mcp_server = span.mcp_server or (span.attributes.get("mcp_server") if span.attributes else None) or (span.attributes.get("server_name") if span.attributes else None)
        is_mcp = span.is_mcp or bool(mcp_server) or span.step_type in ["mcp_tool", "mcp_call", "mcp"]
        exp_mcp_server = getattr(test_case, "expected_mcp_server", None)

        captured_call = CapturedToolCall(
            tool_name=tool_name,
            sequence_position=idx + 1,
            expected_tool=exp_tool,
            arguments=parsed_args,
            expected_arguments=exp_args,
            execution_status=status,
            response=span.output_data or "",
            latency=round(lat, 2),
            retry_count=retry_count,
            error=err_msg,
            is_unnecessary=is_unnec,
            unnecessary_reason=unnec_reason,
            is_mcp=is_mcp,
            mcp_server=mcp_server,
            expected_mcp_server=exp_mcp_server,
            span_id=span.span_id,
            timestamp=span.start_time or span.timestamp,
            metadata=span.attributes or {},
        )
        captured_calls.append(captured_call)

    # Build overall summary
    total_calls = len(captured_calls)
    successful_calls = sum(1 for c in captured_calls if c.execution_status == "success")
    failed_calls = total_calls - successful_calls
    unnec_count = sum(1 for c in captured_calls if c.is_unnecessary)
    total_lat = sum(c.latency for c in captured_calls)
    mcp_calls = sum(1 for c in captured_calls if c.is_mcp)
    mcp_servers = list(dict.fromkeys([c.mcp_server for c in captured_calls if c.mcp_server]))

    # Tool selection accuracy evaluation check
    if not expected_tools:
        selection_accurate = len(captured_calls) == 0
        sel_reason = "No tools expected and zero tools called." if selection_accurate else f"Expected no tools, but called: {tools_called_names}"
    else:
        # Check if actual tools match expected tools
        matched = [t for t in expected_tools if t in tools_called_names]
        forbidden_called = [t for t in (test_case.forbidden_tools if test_case else []) if t in tools_called_names]

        if forbidden_called:
            selection_accurate = False
            sel_reason = f"Forbidden tool(s) called: {forbidden_called}"
        elif len(expected_tools) == 1 and total_calls == 1 and tools_called_names[0] != expected_tools[0]:
            selection_accurate = False
            sel_reason = f"Wrong tool selected. Expected: {expected_tools[0]}, Actual: {tools_called_names[0]}"
        elif len(matched) == len(expected_tools):
            selection_accurate = True
            sel_reason = f"All expected tools correctly selected: {expected_tools}"
        else:
            selection_accurate = False
            missing = [t for t in expected_tools if t not in tools_called_names]
            sel_reason = f"Missing expected tool(s): {missing}. Actual: {tools_called_names}"

    return ToolExecutionSummary(
        tool_calls=captured_calls,
        total_tool_calls=total_calls,
        successful_tool_calls=successful_calls,
        failed_tool_calls=failed_calls,
        unnecessary_tool_calls_count=unnec_count,
        total_tool_latency_ms=round(total_lat, 2),
        actual_tool_sequence=actual_sequence,
        expected_tool_sequence=list(expected_sequence),
        unnecessary_flags=unnecessary_flags,
        retry_loops_detected=max(0, max_consecutive_repeats - 1),
        tools_called_names=tools_called_names,
        is_selection_accurate=selection_accurate,
        selection_reason=sel_reason,
        mcp_tool_calls_count=mcp_calls,
        mcp_servers_involved=mcp_servers,
    )


# ===========================================================================
# 1. Tool Selection Accuracy Evaluator
# ===========================================================================
class ToolSelectionAccuracyEvaluator(BaseEvaluator):
    """Evaluates whether the agent selected the expected tool(s) and avoided forbidden tools.

    Example:
      Expected: search_knowledge_base
      Actual: web_search
      Result: FAILED
      Reason: Wrong tool selected.
    """

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_selection_accuracy",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates whether the agent selected the correct tools and avoided forbidden tools",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        tools_called = summary.tools_called_names
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
                explanation=f"Forbidden tool(s) called: {forbidden_called}.",
                details=f"Forbidden tool(s) called: {forbidden_called}.",
                evidence={
                    "expected": expected_tools,
                    "actual": tools_called,
                    "forbidden_called": forbidden_called,
                    "tool_calls": [c.to_dict() for c in summary.tool_calls],
                },
                evaluator_type=self.evaluator_type,
            )

        if not expected_tools:
            if not tools_called:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    explanation="No specific tools expected; zero tools called.",
                    details="No specific tools expected; zero tools called.",
                    evidence={"expected": [], "actual": [], "tool_calls": []},
                    evaluator_type=self.evaluator_type,
                )
            else:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    threshold=self.threshold,
                    explanation=f"Tools called: {tools_called} (no specific tool required).",
                    details=f"Tools called: {tools_called} (no specific tool required).",
                    evidence={"expected": [], "actual": tools_called, "tool_calls": [c.to_dict() for c in summary.tool_calls]},
                    evaluator_type=self.evaluator_type,
                )

        # If 1 tool expected and exactly 1 called but mismatch
        if len(expected_tools) == 1 and len(tools_called) == 1 and tools_called[0] != expected_tools[0]:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Wrong tool selected. Expected: {expected_tools[0]}, Actual: {tools_called[0]}.",
                details=f"Wrong tool selected. Expected: {expected_tools[0]}, Actual: {tools_called[0]}.",
                evidence={
                    "expected": expected_tools[0],
                    "actual": tools_called[0],
                    "expected_tools": expected_tools,
                    "tools_called": tools_called,
                    "tool_calls": [c.to_dict() for c in summary.tool_calls],
                },
                evaluator_type=self.evaluator_type,
            )

        # Multi-tool recall calculation
        matched = [t for t in expected_tools if t in tools_called]
        missing = [t for t in expected_tools if t not in tools_called]
        score = len(matched) / len(expected_tools)
        passed = score >= self.threshold

        if passed:
            explanation = f"Matched {len(matched)}/{len(expected_tools)} expected tools."
        else:
            explanation = f"Wrong tool selection. Missing expected tool(s): {missing}. Actual calls: {tools_called or 'none'}."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "expected": expected_tools,
                "actual": tools_called,
                "matched": matched,
                "missing": missing,
                "tool_calls": [c.to_dict() for c in summary.tool_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 2. Tool Argument Correctness Evaluator
# ===========================================================================
class ToolArgumentCorrectnessEvaluator(BaseEvaluator):
    """Evaluates whether tool call arguments are well-formed, non-empty, and match expected arguments."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_argument_correctness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Validates tool call input arguments are non-empty, well-formed, and match expected parameters",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        if not summary.tool_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No tool calls made in trace.",
                details="No tool calls made in trace.",
                evidence={"tool_calls_count": 0},
                evaluator_type=self.evaluator_type,
            )

        total_calls = len(summary.tool_calls)
        correct_count = 0
        issues = []

        for call in summary.tool_calls:
            args = call.arguments
            is_parameterless = (
                call.tool_name in ["get_current_date", "now", "date", "get_date"]
                or call.expected_arguments == {}
                or (isinstance(call.expected_arguments, dict) and len(call.expected_arguments) == 0)
            )

            # Check 1: Non-empty or parameterless check
            if is_parameterless:
                is_valid_format = True
            elif isinstance(args, dict):
                is_valid_format = len(args) > 0
            elif isinstance(args, str):
                is_valid_format = bool(args.strip() and args.strip() != "{}")
            elif args is not None:
                is_valid_format = True
            else:
                is_valid_format = False

            if not is_valid_format:
                issues.append(f"Position {call.sequence_position} ({call.tool_name}): Empty or blank arguments.")
                continue

            # Check 2: Expected arguments comparison if specified
            if call.expected_arguments:
                exp = call.expected_arguments
                if isinstance(exp, dict) and isinstance(args, dict):
                    # Check if all expected keys match
                    mismatched_keys = []
                    for k, expected_v in exp.items():
                        if k not in args:
                            mismatched_keys.append(f"Missing key '{k}'")
                        elif str(args[k]).lower().strip() != str(expected_v).lower().strip():
                            # Partial substring fallback
                            if str(expected_v).lower().strip() not in str(args[k]).lower().strip():
                                mismatched_keys.append(f"Key '{k}': expected '{expected_v}', got '{args[k]}'")

                    if mismatched_keys:
                        issues.append(f"Position {call.sequence_position} ({call.tool_name}): {'; '.join(mismatched_keys)}")
                        continue
                elif str(exp).lower().strip() not in str(args).lower().strip():
                    issues.append(f"Position {call.sequence_position} ({call.tool_name}): Expected argument '{exp}' not found in actual '{args}'")
                    continue

            correct_count += 1

        score = correct_count / total_calls if total_calls > 0 else 1.0
        passed = score >= self.threshold

        if passed:
            explanation = f"All {total_calls} tool call arguments were correct and well-formed."
        else:
            explanation = f"{total_calls - correct_count}/{total_calls} tool calls had argument defects: {'; '.join(issues[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "correct_count": correct_count,
                "total_calls": total_calls,
                "issues": issues,
                "tool_calls": [c.to_dict() for c in summary.tool_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 3. Tool Execution Success Evaluator
# ===========================================================================
class ToolExecutionSuccessEvaluator(BaseEvaluator):
    """Evaluates whether all executed tool calls succeeded without runtime exceptions or errors."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_execution_success",
            threshold=threshold,
            weight=weight,
            evaluator_type="reliability",
            description="Verifies that all tool invocations completed successfully without errors",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        if not summary.tool_calls:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No tool calls executed; zero execution errors.",
                details="No tool calls executed; zero execution errors.",
                evidence={"total_calls": 0, "failed_calls": 0},
                evaluator_type=self.evaluator_type,
            )

        failed_calls = [c for c in summary.tool_calls if c.execution_status != "success" or c.error]
        total_calls = len(summary.tool_calls)
        success_count = total_calls - len(failed_calls)

        score = success_count / total_calls if total_calls > 0 else 1.0
        passed = len(failed_calls) == 0

        if passed:
            explanation = f"All {total_calls} tool executions succeeded without error."
        else:
            err_details = [f"{c.tool_name} (pos {c.sequence_position}): {c.error or 'failed status'}" for c in failed_calls]
            explanation = f"{len(failed_calls)}/{total_calls} tool executions failed: {'; '.join(err_details[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_calls": total_calls,
                "successful_calls": success_count,
                "failed_calls": len(failed_calls),
                "errors": [c.error for c in failed_calls if c.error],
                "tool_calls": [c.to_dict() for c in summary.tool_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 4. Tool Efficiency Evaluator
# ===========================================================================
class ToolEfficiencyEvaluator(BaseEvaluator):
    """Measures the efficiency of tool usage, penalizing superfluous calls, high latency, and budget violations."""

    def __init__(self, threshold: float = 0.8, weight: float = 1.0):
        super().__init__(
            name="tool_efficiency",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Measures tool efficiency ratio and optimal tool invocation count",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        if not summary.tool_calls:
            # If no tools expected and none called, efficiency is 100%
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="Optimal execution: no unnecessary tool calls made.",
                details="Optimal execution: no unnecessary tool calls made.",
                evidence={"total_calls": 0, "unnecessary_calls": 0},
                evaluator_type=self.evaluator_type,
            )

        total_calls = summary.total_tool_calls
        unnecessary_calls = summary.unnecessary_tool_calls_count
        expected_count = len(summary.expected_tool_sequence) or (len(test_case.expected_tools) if test_case else 1)

        # Calculate efficiency score: (necessary and successful calls) / total calls
        productive_calls = max(0, total_calls - unnecessary_calls)
        efficiency_ratio = productive_calls / total_calls if total_calls > 0 else 1.0

        # Check max tool call constraints if present
        max_allowed = (
            test_case.max_tool_calls
            if test_case and test_case.max_tool_calls
            else (expected_count + 1 if expected_count > 0 else 2)
        )
        if total_calls > max_allowed:
            efficiency_ratio *= (max_allowed / total_calls)

        score = max(0.0, min(1.0, efficiency_ratio))
        passed = score >= self.threshold

        if passed:
            explanation = f"Tool efficiency is {score * 100:.1f}% ({productive_calls}/{total_calls} productive calls)."
        else:
            explanation = f"Low tool efficiency ({score * 100:.1f}%). {unnecessary_calls} potentially unnecessary tool calls detected."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "total_tool_calls": total_calls,
                "productive_calls": productive_calls,
                "unnecessary_calls": unnecessary_calls,
                "total_tool_latency_ms": summary.total_tool_latency_ms,
                "efficiency_ratio": round(efficiency_ratio, 3),
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 5. Unnecessary Tool Calls Evaluator
# ===========================================================================
class UnnecessaryToolCallsEvaluator(BaseEvaluator):
    """Detects superfluous, redundant, or unneeded tool calls during agent execution.

    Example:
      Expected: LLM → calculator → final answer
      Actual: LLM → search → search → calculator → LLM → final answer
      Flag: ⚠️ 2 potentially unnecessary tool calls
    """

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="unnecessary_tool_calls",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Detects superfluous, redundant, or unneeded tool calls",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        unnecessary_count = summary.unnecessary_tool_calls_count
        total_calls = summary.total_tool_calls

        if total_calls == 0:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No tool calls made; zero unnecessary calls.",
                details="No tool calls made; zero unnecessary calls.",
                evidence={"unnecessary_count": 0, "total_calls": 0, "flags": []},
                evaluator_type=self.evaluator_type,
            )

        if unnecessary_count == 0:
            score = 1.0
            passed = True
            explanation = f"All {total_calls} tool calls were necessary and relevant."
        else:
            # Score decreases with the ratio of unnecessary calls
            score = max(0.0, 1.0 - (unnecessary_count / total_calls))
            passed = unnecessary_count == 0
            explanation = f"⚠️ {unnecessary_count} potentially unnecessary tool call{'s' if unnecessary_count != 1 else ''} detected ({', '.join([c.tool_name for c in summary.tool_calls if c.is_unnecessary])})."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "unnecessary_count": unnecessary_count,
                "total_calls": total_calls,
                "flags": summary.unnecessary_flags,
                "unnecessary_tools": [c.to_dict() for c in summary.tool_calls if c.is_unnecessary],
                "expected_sequence": summary.expected_tool_sequence,
                "actual_sequence": summary.actual_tool_sequence,
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 6. Tool Sequence Correctness Evaluator
# ===========================================================================
class ToolSequenceCorrectnessEvaluator(BaseEvaluator):
    """Evaluates whether the tools were invoked in the correct order/sequence."""

    def __init__(self, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="tool_sequence_correctness",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Evaluates whether tool calls occurred in the expected chronological sequence",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        actual_seq = summary.actual_tool_sequence
        expected_seq = summary.expected_tool_sequence

        if not expected_seq:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="No specific tool sequence required.",
                details="No specific tool sequence required.",
                evidence={"expected_sequence": [], "actual_sequence": actual_seq},
                evaluator_type=self.evaluator_type,
            )

        if not actual_seq:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Expected tool sequence {expected_seq}, but no tools were called.",
                details=f"Expected tool sequence {expected_seq}, but no tools were called.",
                evidence={"expected_sequence": expected_seq, "actual_sequence": []},
                evaluator_type=self.evaluator_type,
            )

        # Calculate sequence similarity using SequenceMatcher
        matcher = SequenceMatcher(None, expected_seq, actual_seq)
        match_ratio = matcher.ratio()

        # Check exact match or ordered subsequence match
        is_exact = actual_seq == expected_seq

        # Check subsequence alignment
        exp_idx = 0
        for tool in actual_seq:
            if exp_idx < len(expected_seq) and tool == expected_seq[exp_idx]:
                exp_idx += 1
        subsequence_match = (exp_idx == len(expected_seq))

        score = 1.0 if is_exact else (0.8 if subsequence_match else match_ratio)
        passed = score >= self.threshold

        if passed:
            explanation = f"Tool sequence matched expected order ({' ➔ '.join(actual_seq)})."
        else:
            explanation = f"Tool sequence mismatch. Expected: [{' ➔ '.join(expected_seq)}], Actual: [{' ➔ '.join(actual_seq)}]."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "expected_sequence": expected_seq,
                "actual_sequence": actual_seq,
                "match_ratio": round(match_ratio, 3),
                "is_exact_match": is_exact,
                "subsequence_match": subsequence_match,
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 7. Tool Retry Behavior Evaluator
# ===========================================================================
class ToolRetryBehaviorEvaluator(BaseEvaluator):
    """Detects unhealthy retry behavior, duplicate loops, or excessive consecutive identical tool calls."""

    def __init__(self, max_allowed_retries: int = 1, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="retry_behavior",
            threshold=threshold,
            weight=weight,
            evaluator_type="behavioral",
            description="Detects whether agent gets stuck in excessive or repetitive tool retry loops",
        )
        self.max_allowed_retries = max_allowed_retries

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        if len(summary.tool_calls) <= 1:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                explanation="Healthy retry behavior; zero retry loops detected.",
                details="Healthy retry behavior; zero retry loops detected.",
                evidence={"retry_loops_detected": 0, "max_consecutive_retries": 0},
                evaluator_type=self.evaluator_type,
            )

        max_retries = max((c.retry_count for c in summary.tool_calls), default=0)
        retry_calls = [c for c in summary.tool_calls if c.retry_count > 0]

        if max_retries <= self.max_allowed_retries:
            score = 1.0
            passed = True
            explanation = "Normal execution behavior; no excessive retry loops."
        else:
            excess = max_retries - self.max_allowed_retries
            score = max(0.0, 1.0 - (excess * 0.35))
            passed = False
            explanation = f"Detected excessive retry loop: tool called {max_retries + 1} consecutive times with identical inputs."

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "max_consecutive_retries": max_retries,
                "retry_calls_count": len(retry_calls),
                "retry_tools": [c.tool_name for c in retry_calls],
            },
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 8. Composite Tool Evaluation Suite Evaluator
# ===========================================================================
class ToolEvaluationSuiteEvaluator(BaseEvaluator):
    """Comprehensive composite evaluator running all 7 tool dimensions in a single check."""

    def __init__(self, threshold: float = 0.85, weight: float = 1.0):
        super().__init__(
            name="tool_evaluation_overall",
            threshold=threshold,
            weight=weight,
            evaluator_type="deterministic",
            description="Comprehensive composite evaluation of all 7 tool execution dimensions",
        )
        self.evaluators = [
            ToolSelectionAccuracyEvaluator(),
            ToolArgumentCorrectnessEvaluator(),
            ToolExecutionSuccessEvaluator(),
            ToolEfficiencyEvaluator(),
            UnnecessaryToolCallsEvaluator(),
            ToolSequenceCorrectnessEvaluator(),
            ToolRetryBehaviorEvaluator(),
        ]

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        summary = extract_tool_telemetry(trace, test_case)
        dimension_results = {}
        scores = []
        failures = []

        for ev in self.evaluators:
            res = ev.evaluate(test_case=test_case, execution_result=execution_result, trace=trace)
            dimension_results[ev.name] = {
                "score": res.score,
                "passed": res.passed,
                "explanation": res.explanation,
            }
            scores.append(res.score)
            if not res.passed:
                failures.append(f"{ev.name}: {res.explanation}")

        composite_score = sum(scores) / len(scores) if scores else 1.0
        passed = composite_score >= self.threshold and len(failures) == 0

        if passed:
            explanation = f"Comprehensive tool evaluation passed with overall score {composite_score * 100:.1f}%."
        else:
            explanation = f"Tool evaluation failed ({composite_score * 100:.1f}%): {'; '.join(failures[:2])}"

        return EvaluationResult(
            metric_name=self.name,
            score=round(composite_score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            details=explanation,
            evidence={
                "composite_score": round(composite_score, 4),
                "dimension_breakdown": dimension_results,
                "failures": failures,
                "telemetry_summary": summary.to_dict(),
            },
            evaluator_type=self.evaluator_type,
        )


ALL_TOOL_EVALUATORS = [
    ToolSelectionAccuracyEvaluator,
    ToolArgumentCorrectnessEvaluator,
    ToolExecutionSuccessEvaluator,
    ToolEfficiencyEvaluator,
    UnnecessaryToolCallsEvaluator,
    ToolSequenceCorrectnessEvaluator,
    ToolRetryBehaviorEvaluator,
    ToolEvaluationSuiteEvaluator,
]
