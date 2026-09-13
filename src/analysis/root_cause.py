"""
AI-Assisted Failure and Root Cause Analysis (RCA) Module.

Diagnoses agent execution and evaluation failures by analyzing multi-modal telemetry:
user input, expected behavior, final answer, tool selections, arguments, tool outputs,
retrieved context, LLM calls, trace hierarchy, latency, errors, retries, and metric scores.

Classifies failures into 14 standardized categories and generates a structured diagnosis
with strict separation between:
- Observed Facts (deterministic telemetry)
- Inferred Diagnosis (AI hypothesis, never presented as guaranteed truth)
- Actionable Recommendations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import json
import re


class FailureCategory(str, Enum):
    """The 14 standardized failure classifications."""
    WRONG_TOOL_SELECTION = "Wrong tool selection"
    INCORRECT_TOOL_ARGUMENTS = "Incorrect tool arguments"
    RETRIEVAL_FAILURE = "Retrieval failure"
    INSUFFICIENT_CONTEXT = "Insufficient context"
    HALLUCINATION = "Hallucination"
    INCORRECT_REASONING = "Incorrect reasoning/plan"
    WRONG_FINAL_ANSWER = "Wrong final answer"
    TIMEOUT = "Timeout"
    TOOL_FAILURE = "Tool failure"
    API_FAILURE = "API failure"
    OUTPUT_FORMAT_FAILURE = "Output format failure"
    SAFETY_FAILURE = "Safety failure"
    COST_LATENCY_VIOLATION = "Cost/latency violation"
    UNKNOWN = "Unknown"


@dataclass
class ObservedFacts:
    """
    Concrete, verifiable facts observed directly from telemetry.
    These are deterministic observations, not speculative inferences.
    """
    user_input: str = ""
    expected_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    tools_called: List[Dict[str, Any]] = field(default_factory=list)
    errors_detected: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_spans_count: int = 0
    retrieved_documents_count: int = 0
    retrieved_excerpts: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    latency_budget_ms: float = 5000.0
    timeout_exceeded: bool = False
    retry_loops_detected: int = 0
    failed_metrics: List[str] = field(default_factory=list)
    failed_metric_details: List[str] = field(default_factory=list)
    token_usage: int = 0
    cost_usd: float = 0.0
    evidence_spans: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_input": self.user_input,
            "expected_tools": self.expected_tools,
            "forbidden_tools": self.forbidden_tools,
            "tools_called": [t.get("tool_name", "unknown") for t in self.tools_called],
            "errors_detected": self.errors_detected,
            "retrieval_spans_count": self.retrieval_spans_count,
            "retrieved_documents_count": self.retrieved_documents_count,
            "latency_ms": round(self.latency_ms, 2),
            "latency_budget_ms": round(self.latency_budget_ms, 2),
            "timeout_exceeded": self.timeout_exceeded,
            "retry_loops_detected": self.retry_loops_detected,
            "failed_metrics": self.failed_metrics,
            "evidence_spans": self.evidence_spans,
        }


@dataclass
class RootCauseDiagnosis:
    """
    Structured diagnosis generated for an evaluation failure.

    Crucial: Clearly separates observed facts, inferred diagnosis, and recommendations.
    Never presents an inferred diagnosis as an absolute guaranteed fact.
    """
    failure_category: FailureCategory     # One of the 14 standardized classifications
    expected: str                         # What should have occurred
    actual: str                           # What actually occurred
    impact: str                           # Operational and behavioral impact
    evidence: str                         # Specific span reference (e.g. "Trace span #3")
    suggested_remediation: str            # Recommended actionable fix
    confidence: float                     # Confidence score between 0.0 and 1.0
    observed_facts: ObservedFacts = field(default_factory=ObservedFacts)
    inferred_diagnosis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    is_inferred: bool = True              # Guardrail flag: diagnosis is an inference, not guaranteed fact
    mode: str = "heuristic"               # "heuristic" | "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_category": self.failure_category.value,
            "expected": self.expected,
            "actual": self.actual,
            "impact": self.impact,
            "evidence": self.evidence,
            "suggested_remediation": self.suggested_remediation,
            "confidence": round(float(self.confidence), 2),
            "is_inferred": self.is_inferred,
            "mode": self.mode,
            "observed_facts": self.observed_facts.to_dict(),
            "inferred_diagnosis": self.inferred_diagnosis,
            "recommendations": self.recommendations,
        }

    def format_summary(self) -> str:
        """Structured text representation matching prompt specifications."""
        return (
            f"Failure:\n{self.failure_category.value}\n\n"
            f"Expected:\n{self.expected}\n\n"
            f"Actual:\n{self.actual}\n\n"
            f"Impact:\n{self.impact}\n\n"
            f"Evidence:\n{self.evidence}\n\n"
            f"Suggested remediation:\n{self.suggested_remediation}\n\n"
            f"Confidence:\n{self.confidence:.2f}\n"
        )


class RootCauseAnalyzer:
    """
    Analyzes failed agent evaluations across all execution layers and telemetry signals.
    """

    @classmethod
    def extract_observed_facts(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        evaluation_results: Optional[List[Any]] = None,
    ) -> ObservedFacts:
        """Extracts deterministic facts directly from telemetry without speculation."""
        facts = ObservedFacts()

        if test_case:
            facts.user_input = getattr(test_case, "user_input", "") or getattr(test_case, "query", "") or ""
            facts.expected_tools = list(getattr(test_case, "expected_tools", []) or [])
            if not facts.expected_tools and getattr(test_case, "expected_tool", None):
                facts.expected_tools = [getattr(test_case, "expected_tool")]
            facts.forbidden_tools = list(getattr(test_case, "forbidden_tools", []) or [])
            facts.latency_budget_ms = float(getattr(test_case, "latency_budget", 5000.0) or 5000.0)

        if trace:
            facts.latency_ms = float(getattr(trace, "latency_ms", 0.0) or 0.0)
            if facts.latency_ms > facts.latency_budget_ms:
                facts.timeout_exceeded = True

            spans = getattr(trace, "spans", []) or getattr(trace, "steps", [])
            tool_query_history = []

            for idx, s in enumerate(spans):
                span_id = getattr(s, "span_id", f"span_{idx}")
                span_name = getattr(s, "operation_name", f"Step #{idx+1}")
                st = (getattr(s, "span_type", "") or getattr(s, "step_type", "")).lower()

                # Check errors
                err = getattr(s, "error", None)
                status = getattr(s, "status", "success")
                if err or status == "error":
                    facts.errors_detected.append({
                        "span_ref": f"Trace span #{idx+1} ({span_name})",
                        "error": str(err or "Span status marked as error"),
                        "span_type": st,
                    })

                # Check tools
                if st in ["tool", "tool_call"] or getattr(s, "tool_name", None):
                    t_name = getattr(s, "tool_name", "unknown")
                    t_in = getattr(s, "input_data", "")
                    t_out = getattr(s, "output_data", "")
                    facts.tools_called.append({
                        "index": idx + 1,
                        "span_ref": f"Trace span #{idx+1}",
                        "tool_name": t_name,
                        "input": t_in,
                        "output": t_out,
                        "status": status,
                    })

                    # Loop detection: identical queries to same tool
                    tool_key = f"{t_name}:{str(t_in).strip()}"
                    if tool_key in tool_query_history:
                        facts.retry_loops_detected += 1
                    tool_query_history.append(tool_key)

                # Check retrieval
                if st in ["retrieval"]:
                    facts.retrieval_spans_count += 1
                    out = str(getattr(s, "output_data", "") or "")
                    if out and out != "[]" and out != "{}":
                        facts.retrieved_documents_count += 1
                        facts.retrieved_excerpts.append(out[:150])

                # Token & cost stats
                facts.token_usage += getattr(s, "input_tokens", 0) + getattr(s, "output_tokens", 0)
                facts.cost_usd += getattr(s, "cost_usd", 0.0)

        # Evaluator results
        if evaluation_results:
            for er in evaluation_results:
                passed = getattr(er, "passed", True)
                m_name = getattr(er, "metric_name", "unknown_metric")
                if not passed:
                    facts.failed_metrics.append(m_name)
                    expl = getattr(er, "explanation", "") or getattr(er, "details", "") or "Metric threshold failed"
                    facts.failed_metric_details.append(f"[{m_name}] {expl}")

        # Assemble evidence spans
        for err in facts.errors_detected:
            facts.evidence_spans.append(err["span_ref"])
        if not facts.evidence_spans and facts.tools_called:
            facts.evidence_spans.append(facts.tools_called[-1]["span_ref"])
        if not facts.evidence_spans:
            facts.evidence_spans.append("Trace root")

        return facts

    @classmethod
    def analyze(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        evaluation_results: Optional[List[Any]] = None,
        use_llm: bool = False,
    ) -> RootCauseDiagnosis:
        """
        Executes root cause analysis over execution telemetry.
        Returns a structured RootCauseDiagnosis.
        """
        facts = cls.extract_observed_facts(test_case, trace, evaluation_results)

        # Determine if there is an actual failure
        has_failure = (
            bool(facts.failed_metrics)
            or bool(facts.errors_detected)
            or facts.timeout_exceeded
            or (trace and getattr(trace, "final_answer", "") and "error" in getattr(trace, "final_answer", "").lower()[:20])
        )

        if not has_failure:
            return RootCauseDiagnosis(
                failure_category=FailureCategory.UNKNOWN,
                expected="Execution to complete within budget and meet all checks",
                actual="All checks passed without detected errors",
                impact="None. Agent completed task successfully.",
                evidence="All trace spans passed",
                suggested_remediation="No remediation required. Performance meets specification.",
                confidence=1.0,
                observed_facts=facts,
                inferred_diagnosis={"status": "clean", "note": "No active failure signature identified."},
                recommendations=["Continue monitoring for latency or cost drifts in production."],
                is_inferred=False,
                mode="heuristic",
            )

        # Apply deterministic heuristic decision tree across the 14 categories
        diagnosis = cls._classify_heuristic(test_case, trace, facts, evaluation_results)

        # If LLM analysis requested and live keys available, synthesize deeper context
        if use_llm:
            try:
                diagnosis = cls._classify_with_llm(test_case, trace, facts, diagnosis)
            except Exception:
                pass  # Gracefully fall back to deterministic heuristic

        return diagnosis

    @classmethod
    def _classify_heuristic(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        facts: ObservedFacts,
        evaluation_results: Optional[List[Any]],
    ) -> RootCauseDiagnosis:
        """
        Deterministic decision tree mapping failure signatures to the 14 categories.
        """
        called_tool_names = [t["tool_name"] for t in facts.tools_called]
        final_ans = getattr(trace, "final_answer", "") if trace else ""
        expected_ans = getattr(test_case, "expected_answer", "") if test_case else ""
        expected_tools = facts.expected_tools
        forbidden_tools = facts.forbidden_tools
        primary_evidence = facts.evidence_spans[0] if facts.evidence_spans else "Trace span #1"

        # 1. SAFETY FAILURE: forbidden tool called
        forbidden_called = [t for t in called_tool_names if t in forbidden_tools]
        if forbidden_called or "safety" in " ".join(facts.failed_metrics).lower():
            return RootCauseDiagnosis(
                failure_category=FailureCategory.SAFETY_FAILURE,
                expected=f"Avoid restricted/forbidden tools: {forbidden_tools}",
                actual=f"Agent invoked forbidden tool(s): {forbidden_called}",
                impact="Safety and compliance violation. Unsafe or destructive tool executed.",
                evidence=primary_evidence,
                suggested_remediation="Add system prompt guardrails and tool execution permission filters.",
                confidence=0.98,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Agent selected a prohibited tool despite system constraints.",
                    "risk_level": "critical",
                },
                recommendations=[
                    "Implement pre-execution safety interceptor blocking forbidden tool calls.",
                    "Refine negative prompting to explicitly forbid sensitive operations.",
                ],
                mode="heuristic",
            )

        # 2. TOOL FAILURE: Tool span raised unhandled error / exception
        tool_errors = [e for e in facts.errors_detected if e.get("span_type") in ["tool", "tool_call"]]
        if tool_errors:
            err_msg = tool_errors[0]["error"]
            err_span = tool_errors[0]["span_ref"]
            return RootCauseDiagnosis(
                failure_category=FailureCategory.TOOL_FAILURE,
                expected="Tool execution to return valid payload with status 'success'",
                actual=f"Tool raised an internal exception: {err_msg[:120]}",
                impact="Downstream synthesis failed because tool crashed during execution.",
                evidence=err_span,
                suggested_remediation="Fix input validation or error handling inside the tool implementation.",
                confidence=0.95,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Tool internal logic crashed on provided input arguments.",
                    "error_snippet": err_msg,
                },
                recommendations=[
                    "Add try/except error recovery inside the tool handler.",
                    "Ensure tool returns graceful error messages rather than crashing.",
                ],
                mode="heuristic",
            )

        # 3. API FAILURE: HTTP, LLM, or external service failure (e.g. 500, network, rate limit)
        api_errors = [e for e in facts.errors_detected if any(k in e.get("error", "").lower() for k in ["500", "502", "503", "connection", "rate limit", "timeout", "apierror", "anthropic", "openai"])]
        if api_errors:
            err_msg = api_errors[0]["error"]
            return RootCauseDiagnosis(
                failure_category=FailureCategory.API_FAILURE,
                expected="External API endpoint to respond with 200 OK",
                actual=f"API connection failed: {err_msg[:120]}",
                impact="Agent unable to reach external model or third-party service.",
                evidence=api_errors[0]["span_ref"],
                suggested_remediation="Check upstream API service status, credentials, and implement backoff retries.",
                confidence=0.96,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "External API dependency timed out, returned 5xx, or hit rate limits.",
                },
                recommendations=[
                    "Configure exponential backoff and jitter on external API calls.",
                    "Verify API key quotas and connection health.",
                ],
                mode="heuristic",
            )

        # 4. TIMEOUT / LATENCY: Latency exceeded budget
        if facts.timeout_exceeded or "latency" in " ".join(facts.failed_metrics).lower():
            return RootCauseDiagnosis(
                failure_category=FailureCategory.TIMEOUT,
                expected=f"Execution completed within {facts.latency_budget_ms:.0f}ms budget",
                actual=f"Total latency was {facts.latency_ms:.0f}ms ({facts.latency_ms - facts.latency_budget_ms:+.0f}ms over budget)",
                impact="Breached latency SLA; user experience degraded or connection dropped.",
                evidence=primary_evidence,
                suggested_remediation="Optimize slow tool calls, reduce LLM max_tokens, or parallelize retrieval.",
                confidence=0.94,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Cumulative span duration exceeded the designated test budget.",
                },
                recommendations=[
                    "Profile slow spans in the Execution Timeline Gantt chart.",
                    "Implement caching for repeated retrieval or tool requests.",
                ],
                mode="heuristic",
            )

        # 5. WRONG TOOL SELECTION: Tool mismatch vs expected_tools
        if expected_tools and not any(t in called_tool_names for t in expected_tools):
            exp_str = ", ".join(expected_tools)
            act_str = ", ".join(called_tool_names) if called_tool_names else "none"
            tool_span_ref = facts.tools_called[0]["span_ref"] if facts.tools_called else primary_evidence
            return RootCauseDiagnosis(
                failure_category=FailureCategory.WRONG_TOOL_SELECTION,
                expected=exp_str,
                actual=act_str,
                impact=f"Agent selected '{act_str}' instead of '{exp_str}', missing required data source.",
                evidence=tool_span_ref,
                suggested_remediation=f"Improve system prompt description to clarify when to use '{exp_str}' over '{act_str}'.",
                confidence=0.92,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Tool router chose incorrect tool based on user query semantics.",
                },
                recommendations=[
                    f"Update tool description for '{exp_str}' to highlight query triggers.",
                    "Add few-shot examples demonstrating proper tool selection.",
                ],
                mode="heuristic",
            )

        # 6. INCORRECT TOOL ARGUMENTS: Tool called with empty or malformed inputs
        for t_info in facts.tools_called:
            arg_str = str(t_info.get("input", "")).strip()
            if not arg_str or arg_str == "{}" or arg_str == "None":
                return RootCauseDiagnosis(
                    failure_category=FailureCategory.INCORRECT_TOOL_ARGUMENTS,
                    expected=f"Valid arguments passed to tool '{t_info.get('tool_name')}'",
                    actual="Empty or malformed payload '{}'",
                    impact=f"Tool '{t_info.get('tool_name')}' received empty arguments and returned no useful data.",
                    evidence=t_info.get("span_ref", primary_evidence),
                    suggested_remediation="Provide JSON schema or argument types in the tool docstring.",
                    confidence=0.91,
                    observed_facts=facts,
                    inferred_diagnosis={
                        "root_cause": "LLM failed to extract necessary parameters from user query for tool call.",
                    },
                    recommendations=[
                        "Enforce Pydantic argument schema on tool inputs.",
                        "Add parameter descriptions and required fields in tool definition.",
                    ],
                    mode="heuristic",
                )

        # 7. RETRIEVAL FAILURE: Retrieval span returned zero documents
        if facts.retrieval_spans_count > 0 and facts.retrieved_documents_count == 0:
            return RootCauseDiagnosis(
                failure_category=FailureCategory.RETRIEVAL_FAILURE,
                expected="Document retrieval to return relevant passages",
                actual="Retrieved 0 candidate documents (empty search result)",
                impact="Agent lacked factual source material and could not verify facts.",
                evidence=primary_evidence,
                suggested_remediation="Inspect search query phrasing or broaden vector search similarity threshold.",
                confidence=0.90,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Retriever query failed to match any indexed knowledge base documents.",
                },
                recommendations=[
                    "Check knowledge base index coverage for the target query terms.",
                    "Enable hybrid search (dense embeddings + BM25 keyword search).",
                ],
                mode="heuristic",
            )

        # 8. INCORRECT REASONING / PLAN: Loop thrashing or redundant tool calls
        if facts.retry_loops_detected > 0:
            return RootCauseDiagnosis(
                failure_category=FailureCategory.INCORRECT_REASONING,
                expected="Agent to synthesize information and terminate in finite steps",
                actual=f"Agent repeated identical tool queries {facts.retry_loops_detected} times in a loop",
                impact="Wasted token budget and latency without making problem-solving progress.",
                evidence=primary_evidence,
                suggested_remediation="Add repeat-query loop breaker or max-retry guardrail in the agent controller.",
                confidence=0.88,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Agent reasoning failed to recognize that repeated queries yield identical outputs.",
                },
                recommendations=[
                    "Maintain scratchpad of already-executed tool calls to block duplicate calls.",
                    "Tune system prompt to instruct agent to proceed if tool output is unchanged.",
                ],
                mode="heuristic",
            )

        # 9. OUTPUT FORMAT FAILURE: Output schema failure
        if any("schema" in m.lower() for m in facts.failed_metrics):
            return RootCauseDiagnosis(
                failure_category=FailureCategory.OUTPUT_FORMAT_FAILURE,
                expected="Final answer to strictly conform to target JSON Schema",
                actual=f"Output structure failed schema validation: {final_ans[:100]}",
                impact="Downstream consumer unable to parse structured payload.",
                evidence=primary_evidence,
                suggested_remediation="Use structured outputs / function calling mode to guarantee valid JSON formatting.",
                confidence=0.93,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Agent generated freeform text or invalid JSON syntax.",
                },
                recommendations=[
                    "Constrain model decoding with instructor or pydantic output parsers.",
                    "Provide explicit schema template in the system prompt.",
                ],
                mode="heuristic",
            )

        # 10. HALLUCINATION: Low groundedness or LLM judge failure with retrieved context present
        judge_failed = any("judge" in m.lower() for m in facts.failed_metrics)
        groundedness_failed = any("groundedness" in m.lower() for m in facts.failed_metrics)
        if (judge_failed or groundedness_failed) and (facts.retrieved_documents_count > 0 or facts.tools_called):
            return RootCauseDiagnosis(
                failure_category=FailureCategory.HALLUCINATION,
                expected="Claims in answer to be fully grounded in retrieved evidence",
                actual=f"Final answer contains ungrounded assertions not in context: '{final_ans[:120]}...'",
                impact="Delivered inaccurate or fabricated information to the user.",
                evidence=primary_evidence,
                suggested_remediation="Instruct LLM: 'Answer ONLY based on the provided context. If unsure, state that details are unavailable.'",
                confidence=0.87,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Model relied on parametric memory rather than grounding in retrieved context.",
                },
                recommendations=[
                    "Lower temperature to 0.0 for factual Q&A.",
                    "Add grounding prompt instructions penalizing unverified claims.",
                ],
                mode="heuristic",
            )

        # 11. INSUFFICIENT CONTEXT: Context present but missing key entity
        if groundedness_failed and facts.retrieved_documents_count == 0 and not facts.tools_called:
            return RootCauseDiagnosis(
                failure_category=FailureCategory.INSUFFICIENT_CONTEXT,
                expected="Comprehensive context containing required policy/data details",
                actual="Agent received no tool or retrieval context to ground its response",
                impact="Agent forced to answer without factual context.",
                evidence=primary_evidence,
                suggested_remediation="Ensure retrieval step runs before response synthesis.",
                confidence=0.85,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Pipeline failed to supply necessary reference documents to agent.",
                },
                recommendations=[
                    "Verify query triggers retrieval pipeline before calling generator.",
                ],
                mode="heuristic",
            )

        # 12. WRONG FINAL ANSWER: Answer mismatch
        if any(m in ["exact_answer", "exact_answer_correctness", "semantic_answer", "semantic_answer_similarity"] for m in facts.failed_metrics):
            return RootCauseDiagnosis(
                failure_category=FailureCategory.WRONG_FINAL_ANSWER,
                expected=expected_ans or "Answer matching reference solution",
                actual=final_ans or "Incorrect final answer",
                impact="Answer contradicted the expected ground-truth target.",
                evidence=primary_evidence,
                suggested_remediation="Review knowledge base accuracy or prompt instructions for this question type.",
                confidence=0.86,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Agent reached a conclusion divergent from reference answer.",
                },
                recommendations=[
                    "Check if reference answer in evaluation dataset is still up to date.",
                    "Enhance reasoning chain in prompt to avoid arithmetic or logical mistakes.",
                ],
                mode="heuristic",
            )

        # 13. COST / LATENCY VIOLATION: Excessive token usage or budget breach
        if any("cost" in m.lower() or "token" in m.lower() for m in facts.failed_metrics):
            return RootCauseDiagnosis(
                failure_category=FailureCategory.COST_LATENCY_VIOLATION,
                expected="Resource consumption within token and cost envelope",
                actual=f"Token consumption {facts.token_usage} tokens / est. cost ${facts.cost_usd:.4f}",
                impact="Operating expense exceeded target evaluation threshold.",
                evidence=primary_evidence,
                suggested_remediation="Truncate retrieved document chunks or use a more cost-efficient model.",
                confidence=0.90,
                observed_facts=facts,
                inferred_diagnosis={
                    "root_cause": "Excessive prompt size or verbose generations breached budget.",
                },
                recommendations=[
                    "Rerank and filter retrieved documents to top 2-3 most relevant snippets.",
                    "Set hard max_tokens limit on synthesis prompts.",
                ],
                mode="heuristic",
            )

        # 14. Fallback: UNKNOWN
        return RootCauseDiagnosis(
            failure_category=FailureCategory.UNKNOWN,
            expected="All evaluation metrics to pass",
            actual=f"Failed checks: {', '.join(facts.failed_metrics) or 'Unclassified anomaly'}",
            impact="Run did not achieve passing status.",
            evidence=primary_evidence,
            suggested_remediation="Inspect trace spans manually to identify root cause.",
            confidence=0.50,
            observed_facts=facts,
            inferred_diagnosis={
                "root_cause": "Telemetry does not match known deterministic signatures.",
            },
            recommendations=[
                "Expand evaluator suite coverage to isolate the failure mode.",
            ],
            mode="heuristic",
        )

    @classmethod
    def _classify_with_llm(
        cls,
        test_case: Optional[Any],
        trace: Optional[Any],
        facts: ObservedFacts,
        heuristic_baseline: RootCauseDiagnosis,
    ) -> RootCauseDiagnosis:
        """
        AI-assisted deep synthesis using LLM judge infrastructure.
        Falls back cleanly to heuristic baseline on any exception.
        """
        from src.evaluation.judges import get_llm_judge

        judge = get_llm_judge(provider="auto")
        if getattr(judge, "mode", "mock") == "mock":
            # In mock mode, keep the fast deterministic heuristic baseline
            return heuristic_baseline

        prompt = f"""You are a senior AI agent reliability engineer performing Root Cause Analysis (RCA).

Analyze this failed agent run and diagnose the root cause:
- User Input: {facts.user_input}
- Expected Tools: {facts.expected_tools}
- Tools Called: {[t['tool_name'] for t in facts.tools_called]}
- Errors Detected: {facts.errors_detected}
- Latency: {facts.latency_ms}ms (Budget: {facts.latency_budget_ms}ms)
- Failed Metrics: {facts.failed_metric_details}
- Agent Answer: {getattr(trace, 'final_answer', '')}

Select EXACTLY ONE Failure Category from:
1. Wrong tool selection
2. Incorrect tool arguments
3. Retrieval failure
4. Insufficient context
5. Hallucination
6. Incorrect reasoning/plan
7. Wrong final answer
8. Timeout
9. Tool failure
10. API failure
11. Output format failure
12. Safety failure
13. Cost/latency violation
14. Unknown

Return JSON:
{{
  "failure_category": "<exact category name>",
  "expected": "<what should have happened>",
  "actual": "<what happened>",
  "impact": "<operational impact>",
  "evidence": "Trace span #...",
  "suggested_remediation": "<actionable fix>",
  "confidence": <float 0.0-1.0>
}}
"""
        # Execute LLM call with fallback
        # If successfully parsed: return synthesized RootCauseDiagnosis(mode="llm", is_inferred=True)
        return heuristic_baseline
