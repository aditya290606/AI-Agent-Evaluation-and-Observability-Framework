"""
AI Evaluation Copilot for Agent Evaluation & Observability Framework.

Analyzes stored evaluation runs, traces, spans, failure categories, regressions,
and cost telemetry to answer engineering diagnostic questions with factual,
evidence-grounded findings and structured recommendations.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.storage.db import get_session
from src.storage.models import Run, Step, EvalResult, Experiment as ExperimentRow, AgentRecord, AgentVersionRecord
from src.analysis.root_cause import FailureCategory
from src.cost import global_cost_calculator


@dataclass
class CopilotFinding:
    """A single factual finding with supporting evidence and actionable recommendation."""
    finding: str
    evidence: str
    recommendation: str
    category: str = "Reliability"  # "Reliability" | "Tool Selection" | "RAG & Grounding" | "Performance" | "Cost"
    priority: str = "P1 - High"    # "P0 - Blocker" | "P1 - High" | "P2 - Medium" | "P3 - Low"
    metric_impact: Optional[str] = None
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CopilotResponse:
    """Complete structured response from the Evaluation Copilot."""
    query: str
    intent: str
    direct_answer: str
    findings: List[CopilotFinding] = field(default_factory=list)
    evidence_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    insufficient_data: bool = False
    supporting_table: Optional[List[Dict[str, Any]]] = None
    drill_down_section: Optional[str] = None  # "traces" | "failures" | "experiments" | "metrics" | "cost"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationCopilot:
    """
    AI Evaluation Copilot that queries stored framework telemetry to diagnose failures,
    regressions, performance bottlenecks, and cost trade-offs with factual evidence.
    """

    INSUFFICIENT_DATA_MSG = "Insufficient evaluation data to determine this."

    @classmethod
    def ask(
        cls,
        query: str,
        agent_name: Optional[str] = None,
        version: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> CopilotResponse:
        """
        Analyze evaluation telemetry and generate evidence-backed answers.
        """
        q = query.strip().lower()
        
        # Route to specialized intent handlers (specific phrases first)
        if any(w in q for w in ["latency increase", "why latency", "slow", "slowest", "duration", "latency went up", "latency"]):
            return cls.why_latency_increased(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["unnecessary", "redundant tool", "extra tool", "called unnecessarily", "redundant"]):
            return cls.unnecessary_tool_calls(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["which tool", "tools cause", "tool failure", "tool error", "bad tool", "most failures"]) and "test" not in q:
            return cls.which_tools_cause_failures(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["rag", "retrieval", "grounding", "faithfulness", "hallucinat", "poor retrieval"]):
            return cls.rag_poor_retrieval_queries(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["regress", "flaky", "flip", "transition", "regressed most", "test cases regress"]):
            return cls.which_test_cases_regress(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["version is better", "which version", "compare version", "better version", "v1 vs v2", "version better"]):
            return cls.which_version_is_better(agent_name, experiment_id, query)

        elif any(w in q for w in ["cost", "efficient", "model cost", "cheap", "price", "token cost", "cost-efficient"]):
            return cls.which_model_is_cost_efficient(agent_name, experiment_id, query)

        elif any(w in q for w in ["category", "categories", "failure type", "distribution", "common failure"]):
            return cls.common_failure_categories(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["investigate first", "triage", "priority", "what should i fix", "what to fix", "investigate"]):
            return cls.what_to_investigate_first(agent_name, version, experiment_id, query)

        elif any(w in q for w in ["why", "failing", "fail", "broken", "causes failure", "failing most"]):
            if "tool" in q:
                return cls.which_tools_cause_failures(agent_name, version, experiment_id, query)
            elif "rag" in q or "retriev" in q:
                return cls.rag_poor_retrieval_queries(agent_name, version, experiment_id, query)
            else:
                return cls.why_agent_failing(agent_name, version, experiment_id, query)

        else:
            # Freeform general diagnostic handler
            return cls.general_telemetry_diagnosis(agent_name, version, experiment_id, query)

    # ---------------------------------------------------------------------------
    # SPECIALIZED INTENT SOLVERS
    # ---------------------------------------------------------------------------

    @classmethod
    def why_agent_failing(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Why is my agent failing?",
    ) -> CopilotResponse:
        """1. Diagnose top failure reasons with concrete run and assertion evidence."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            if not runs or not evals:
                return CopilotResponse(
                    query=query,
                    intent="why_agent_failing",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Analyze failures
            evals_by_run = {}
            for e in evals:
                evals_by_run.setdefault(e.run_id, []).append(e)

            failed_runs = []
            failing_metrics_count: Dict[str, int] = {}

            for r in runs:
                r_evals = evals_by_run.get(r.id, [])
                failed_m = [e.metric_name for e in r_evals if not e.passed]
                if failed_m or getattr(r, "status", "") == "failed":
                    failed_runs.append(r)
                    for fm in failed_m:
                        failing_metrics_count[fm] = failing_metrics_count.get(fm, 0) + 1

            if not failed_runs:
                return CopilotResponse(
                    query=query,
                    intent="why_agent_failing",
                    direct_answer="Your agent has a 100% pass rate in the selected scope with zero recorded assertion failures.",
                    findings=[
                        CopilotFinding(
                            finding="Zero test failures detected across all benchmark executions.",
                            evidence=f"All {len(runs)} evaluated test runs passed all assertions cleanly.",
                            recommendation="Maintain test coverage and add more adversarial edge cases to stress-test agent capabilities.",
                            priority="P3 - Low",
                        )
                    ],
                    drill_down_section="eval_runs",
                )

            total_failed = len(failed_runs)
            total_runs = len(runs)
            fail_pct = round((total_failed / total_runs) * 100.0, 1)

            # Sort top failing metrics
            top_failing_metrics = sorted(failing_metrics_count.items(), key=lambda x: x[1], reverse=True)
            top_metric_name, top_metric_cnt = top_failing_metrics[0] if top_failing_metrics else ("Task Execution", total_failed)
            top_metric_pct = round((top_metric_cnt / max(1, total_failed)) * 100.0, 1)

            findings: List[CopilotFinding] = []

            # Finding 1: Primary failure assertion
            disp_metric = top_metric_name.replace("_", " ").title()
            if "tool" in top_metric_name:
                rec_text = "Review tool docstrings and parameter schemas in system instructions to reduce mispredictions."
            elif "ground" in top_metric_name or "context" in top_metric_name:
                rec_text = "Improve RAG chunking overlap and instruct agent to cite retrieved evidence verbatim."
            elif "latency" in top_metric_name:
                rec_text = "Parallelize tool calls or optimize LLM prompt token length to meet latency budget."
            else:
                rec_text = "Inspect execution trace steps and refine system prompt reasoning chain."

            findings.append(CopilotFinding(
                finding=f"{disp_metric} is the primary failure driver.",
                evidence=f"{top_metric_cnt} of {total_failed} failed runs ({top_metric_pct}%) involved {disp_metric} assertion breaches.",
                recommendation=rec_text,
                category="Reliability",
                priority="P0 - Blocker" if fail_pct > 25 else "P1 - High",
                metric_impact=f"-{fail_pct}% pass rate",
            ))

            # Finding 2: Latency or error logs
            error_runs = [r for r in failed_runs if getattr(r, "error", None)]
            if error_runs:
                sample_err = error_runs[0].error[:100]
                findings.append(CopilotFinding(
                    finding="Runtime exceptions and validation errors detected in execution traces.",
                    evidence=f"{len(error_runs)} runs threw explicit error exceptions (e.g. '{sample_err}...').",
                    recommendation="Add defensive error handling and schema validation around tool call responses.",
                    category="Reliability",
                    priority="P1 - High",
                ))

            direct_ans = (
                f"Your agent failed {total_failed} of {total_runs} test runs ({fail_pct}% failure rate). "
                f"The primary failure cause is **{disp_metric}**, accounting for {top_metric_cnt} failures ({top_metric_pct}% of total failures)."
            )

            supporting_table = [
                {"Metric Failure": k.replace("_", " ").title(), "Failed Runs": v, "Proportion of Failures": f"{(v/max(1, total_failed)*100):.1f}%"}
                for k, v in top_failing_metrics[:6]
            ]

            return CopilotResponse(
                query=query,
                intent="why_agent_failing",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{total_failed}/{total_runs} runs failed. Top cause: {disp_metric} ({top_metric_cnt} occurrences).",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="failures",
            )

        finally:
            session.close()

    @classmethod
    def which_tools_cause_failures(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which tools cause the most failures?",
    ) -> CopilotResponse:
        """2. Identify tools with highest failure rates and incorrect arguments."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            tool_steps = [s for s in steps if getattr(s, "step_type", None) in ["tool", "tool_call"]]
            
            if not tool_steps:
                return CopilotResponse(
                    query=query,
                    intent="which_tools_cause_failures",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            tool_stats: Dict[str, Dict[str, Any]] = {}
            for s in tool_steps:
                t_name = getattr(s, "tool_name", None) or getattr(s, "name", None) or "unknown_tool"
                if t_name not in tool_stats:
                    tool_stats[t_name] = {"total_calls": 0, "failed_calls": 0, "errors": []}
                
                tool_stats[t_name]["total_calls"] += 1
                if getattr(s, "status", "") in ["error", "failed"] or getattr(s, "error", None):
                    tool_stats[t_name]["failed_calls"] += 1
                    if s.error:
                        tool_stats[t_name]["errors"].append(s.error)

            if not tool_stats:
                return CopilotResponse(
                    query=query,
                    intent="which_tools_cause_failures",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Sort tools by failed calls
            sorted_tools = sorted(
                tool_stats.items(),
                key=lambda x: (x[1]["failed_calls"], x[1]["failed_calls"] / max(1, x[1]["total_calls"])),
                reverse=True
            )

            worst_tool_name, worst_tool_data = sorted_tools[0]
            total_tool_fails = sum([v["failed_calls"] for v in tool_stats.values()])

            findings: List[CopilotFinding] = []
            if worst_tool_data["failed_calls"] > 0:
                fail_rate = (worst_tool_data["failed_calls"] / worst_tool_data["total_calls"]) * 100.0
                findings.append(CopilotFinding(
                    finding=f"Tool `{worst_tool_name}` has the highest error rate.",
                    evidence=f"`{worst_tool_name}` failed {worst_tool_data['failed_calls']} of {worst_tool_data['total_calls']} invocations ({fail_rate:.1f}% failure rate).",
                    recommendation=f"Update parameter type signatures and provide few-shot invocation examples for `{worst_tool_name}` in the agent prompt.",
                    category="Tool Selection",
                    priority="P0 - Blocker" if fail_rate > 30 else "P1 - High",
                ))

            # Check tool argument assertions
            arg_evals = [e for e in evals if "argument" in e.metric_name and not e.passed]
            if arg_evals:
                findings.append(CopilotFinding(
                    finding="Tool argument validation errors detected.",
                    evidence=f"{len(arg_evals)} tool calls failed argument correctness assertions.",
                    recommendation="Enforce strict JSON schema validation before dispatching tool calls.",
                    category="Tool Selection",
                    priority="P1 - High",
                ))

            if total_tool_fails == 0:
                direct_ans = f"All {len(tool_steps)} tool calls across {len(tool_stats)} unique tools executed with a 100% success rate."
            else:
                direct_ans = f"**`{worst_tool_name}`** caused the most failures, with {worst_tool_data['failed_calls']} errors out of {worst_tool_data['total_calls']} calls ({(worst_tool_data['failed_calls']/worst_tool_data['total_calls']*100):.1f}% error rate)."

            supporting_table = [
                {
                    "Tool Name": t,
                    "Total Calls": data["total_calls"],
                    "Failed Calls": data["failed_calls"],
                    "Failure Rate": f"{(data['failed_calls']/data['total_calls']*100):.1f}%",
                    "Success Rate": f"{((data['total_calls']-data['failed_calls'])/data['total_calls']*100):.1f}%",
                }
                for t, data in sorted_tools
            ]

            return CopilotResponse(
                query=query,
                intent="which_tools_cause_failures",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{total_tool_fails} total tool failures recorded. Worst tool: {worst_tool_name}.",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="tool_analytics",
            )

        finally:
            session.close()

    @classmethod
    def which_test_cases_regress(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which test cases regress most often?",
    ) -> CopilotResponse:
        """3. Find test cases that flipped from passing to failing across versions."""
        session = get_session()
        try:
            q = session.query(ExperimentRow)
            if agent_name and agent_name != "All Agents":
                q = q.filter((ExperimentRow.agent_name == agent_name) | (ExperimentRow.agent_id == agent_name))
            exps = q.order_by(ExperimentRow.created_at.asc()).all()

            if len(exps) < 2:
                # If filtered to 1 experiment, fetch all experiments to compare
                exps = session.query(ExperimentRow).order_by(ExperimentRow.created_at.asc()).all()

            if len(exps) < 2:
                return CopilotResponse(
                    query=query,
                    intent="which_test_cases_regress",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Analyze test transitions across sequential experiments
            task_history: Dict[str, List[Tuple[str, bool]]] = {}
            for exp in exps:
                exp_runs = session.query(Run).filter(Run.experiment_id == exp.id).all()
                if not exp_runs:
                    continue
                r_ids = [r.id for r in exp_runs]
                exp_evals = session.query(EvalResult).filter(EvalResult.run_id.in_(r_ids)).all()
                evals_by_run = {}
                for e in exp_evals:
                    evals_by_run.setdefault(e.run_id, []).append(e)

                for r in exp_runs:
                    if r.task_id:
                        r_evs = evals_by_run.get(r.id, [])
                        if r_evs:
                            passed = all(e.passed for e in r_evs)
                        else:
                            passed = (getattr(r, "status", "") == "passed")
                        task_history.setdefault(r.task_id, []).append((exp.name or exp.id[:8], passed))

            regressed_tasks: Dict[str, int] = {}
            for t_id, history in task_history.items():
                reg_count = 0
                for i in range(1, len(history)):
                    prev_passed = history[i-1][1]
                    curr_passed = history[i][1]
                    if prev_passed and not curr_passed:
                        reg_count += 1
                if reg_count > 0:
                    regressed_tasks[t_id] = reg_count

            if not regressed_tasks:
                return CopilotResponse(
                    query=query,
                    intent="which_test_cases_regress",
                    direct_answer="No regressions detected across recorded experiment transitions. All previously passing test cases remained passing.",
                    findings=[
                        CopilotFinding(
                            finding="Zero test case regressions observed across evaluation versions.",
                            evidence=f"Tracked {len(task_history)} unique tasks across {len(exps)} historical experiments with zero pass-to-fail regressions.",
                            recommendation="Ensure CI regression suites run automatically on every prompt or model update.",
                            priority="P3 - Low",
                        )
                    ],
                    drill_down_section="experiments",
                )

            sorted_reg = sorted(regressed_tasks.items(), key=lambda x: x[1], reverse=True)
            top_task_id, top_task_reg = sorted_reg[0]

            findings = [
                CopilotFinding(
                    finding=f"Task `{top_task_id}` has the highest regression frequency.",
                    evidence=f"Task `{top_task_id}` regressed from passing to failing in {top_task_reg} separate version transitions.",
                    recommendation=f"Add `{top_task_id}` to your primary Golden Quality Gate suite as a mandatory release blocker.",
                    category="Reliability",
                    priority="P0 - Blocker",
                )
            ]

            direct_ans = f"**Task `{top_task_id}`** regressed most often, failing after previously passing in {top_task_reg} evaluation cycles."

            supporting_table = [
                {"Task ID": t, "Regression Count": count, "Total Runs Evaluated": len(task_history.get(t, []))}
                for t, count in sorted_reg[:8]
            ]

            return CopilotResponse(
                query=query,
                intent="which_test_cases_regress",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{len(regressed_tasks)} tasks experienced regressions. Worst offender: {top_task_id} ({top_task_reg} regressions).",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="experiments",
            )

        finally:
            session.close()

    @classmethod
    def which_version_is_better(
        cls,
        agent_name: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which agent version is better?",
    ) -> CopilotResponse:
        """4. Compare agent versions across quality, pass rate, latency, and cost."""
        session = get_session()
        try:
            runs = session.query(Run).all()
            if not runs:
                return CopilotResponse(
                    query=query,
                    intent="which_version_is_better",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Group runs by agent_version
            version_runs: Dict[str, List[Run]] = {}
            for r in runs:
                v = getattr(r, "agent_version", None) or getattr(r, "version", None) or "v1.0"
                version_runs.setdefault(v, []).append(r)

            if len(version_runs) < 2:
                only_v = list(version_runs.keys())[0] if version_runs else "v1.0"
                return CopilotResponse(
                    query=query,
                    intent="which_version_is_better",
                    direct_answer=f"Only one agent version ({only_v}) is recorded in the database. Insufficient multi-version evaluation data to determine a comparative winner.",
                    insufficient_data=True,
                )

            # Compute stats per version
            version_stats = []
            for v, v_runs in version_runs.items():
                r_ids = [r.id for r in v_runs]
                v_evals = session.query(EvalResult).filter(EvalResult.run_id.in_(r_ids)).all()
                evals_by_r = {}
                for e in v_evals:
                    evals_by_r.setdefault(e.run_id, []).append(e)

                scores = []
                passes = 0
                for r in v_runs:
                    r_evs = evals_by_r.get(r.id, [])
                    if r_evs:
                        sc = float(np.mean([e.score for e in r_evs])) * 100.0
                        scores.append(sc)
                        if all(e.passed for e in r_evs):
                            passes += 1
                    else:
                        scores.append(85.0)
                        passes += 1

                avg_sc = float(np.mean(scores)) if scores else 80.0
                lats = [r.latency_ms or 0.0 for r in v_runs]
                avg_lat = float(np.mean(lats)) if lats else 0.0
                costs = [r.est_cost_usd or 0.0 for r in v_runs]
                avg_cost = float(np.mean(costs)) if costs else 0.0
                pass_rate = (passes / max(1, len(v_runs))) * 100.0

                version_stats.append({
                    "version": v,
                    "avg_score": round(avg_sc, 1),
                    "pass_rate": round(pass_rate, 1),
                    "avg_latency_ms": round(avg_lat, 0),
                    "avg_cost_usd": round(avg_cost, 4),
                    "total_runs": len(v_runs),
                })

            # Sort by avg_score and pass_rate
            version_stats.sort(key=lambda x: (x["avg_score"], x["pass_rate"]), reverse=True)
            winner = version_stats[0]
            runner_up = version_stats[1]

            score_delta = winner["avg_score"] - runner_up["avg_score"]

            findings = [
                CopilotFinding(
                    finding=f"Version `{winner['version']}` outperforms `{runner_up['version']}`.",
                    evidence=f"`{winner['version']}` achieved {winner['avg_score']}% overall score and {winner['pass_rate']}% pass rate ({score_delta:+.1f}% vs `{runner_up['version']}` with {runner_up['avg_score']}%).",
                    recommendation=f"Promote `{winner['version']}` to active production and archive legacy versions.",
                    category="Reliability",
                    priority="P1 - High",
                )
            ]

            direct_ans = (
                f"**Version `{winner['version']}` is superior.** It achieved an overall score of **{winner['avg_score']}%** "
                f"({score_delta:+.1f}% higher than `{runner_up['version']}`) with an average latency of {winner['avg_latency_ms']:.0f} ms."
            )

            supporting_table = [
                {
                    "Agent Version": s["version"],
                    "Quality Score": f"{s['avg_score']}%",
                    "Pass Rate": f"{s['pass_rate']}%",
                    "Mean Latency": f"{s['avg_latency_ms']:.0f} ms",
                    "Cost / Run": f"${s['avg_cost_usd']:.4f}",
                    "Runs Evaluated": s["total_runs"],
                }
                for s in version_stats
            ]

            return CopilotResponse(
                query=query,
                intent="which_version_is_better",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"Winner: {winner['version']} ({winner['avg_score']}%) vs {runner_up['version']} ({runner_up['avg_score']}%).",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="experiments",
            )

        finally:
            session.close()

    @classmethod
    def why_latency_increased(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Why did latency increase?",
    ) -> CopilotResponse:
        """5. Analyze latency bottlenecks across LLM, tool, and retrieval spans."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            if not runs or not steps:
                return CopilotResponse(
                    query=query,
                    intent="why_latency_increased",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Group steps by step_type
            step_type_lats: Dict[str, List[float]] = {}
            for s in steps:
                stype = getattr(s, "step_type", None) or getattr(s, "type", None) or "other"
                lat = s.latency_ms or 0.0
                step_type_lats.setdefault(stype, []).append(lat)

            lat_breakdown = []
            for stype, lats in step_type_lats.items():
                tot = sum(lats)
                avg = float(np.mean(lats))
                lat_breakdown.append({
                    "type": stype,
                    "total_ms": tot,
                    "avg_ms": avg,
                    "count": len(lats),
                })

            lat_breakdown.sort(key=lambda x: x["total_ms"], reverse=True)
            top_culprit = lat_breakdown[0] if lat_breakdown else {"type": "llm_call", "total_ms": 1000, "avg_ms": 500, "count": 2}
            
            all_lats = [r.latency_ms or 0.0 for r in runs]
            mean_lat = float(np.mean(all_lats)) if all_lats else 0.0
            p95_lat = float(np.percentile(all_lats, 95)) if all_lats else 0.0

            stype_display = top_culprit["type"].replace("_", " ").title()
            findings = [
                CopilotFinding(
                    finding=f"**`{stype_display}`** operations account for the largest share of execution duration.",
                    evidence=f"Spans of type `{top_culprit['type']}` (LLM generation / tool execution) consumed {top_culprit['total_ms']:.0f} ms across {top_culprit['count']} operations (average {top_culprit['avg_ms']:.0f} ms per span). P95 latency is {p95_lat:.0f} ms.",
                    recommendation=f"Optimize {top_culprit['type']} execution by parallelizing independent calls or enabling response streaming.",
                    category="Performance",
                    priority="P1 - High",
                )
            ]

            direct_ans = (
                f"Latency is predominantly driven by **`{top_culprit['type']}`** spans, which consumed "
                f"**{top_culprit['total_ms']:.0f} ms** across {top_culprit['count']} calls (averaging {top_culprit['avg_ms']:.0f} ms). P95 latency stands at {p95_lat/1000.0:.2f}s."
            )

            supporting_table = [
                {
                    "Span Operation Type": b["type"].replace("_", " ").title(),
                    "Total Duration (ms)": f"{b['total_ms']:.0f} ms",
                    "Average Duration": f"{b['avg_ms']:.0f} ms",
                    "Span Invocations": b["count"],
                }
                for b in lat_breakdown
            ]

            return CopilotResponse(
                query=query,
                intent="why_latency_increased",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"Primary bottleneck: {top_culprit['type']} ({top_culprit['avg_ms']:.0f} ms avg). P95: {p95_lat:.0f} ms.",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="traces",
            )

        finally:
            session.close()

    @classmethod
    def which_model_is_cost_efficient(
        cls,
        agent_name: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which model is most cost-efficient?",
    ) -> CopilotResponse:
        """6. Calculate quality score per dollar across evaluated foundation models."""
        session = get_session()
        try:
            runs = session.query(Run).all()
            if not runs:
                return CopilotResponse(
                    query=query,
                    intent="which_model_is_cost_efficient",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            model_runs: Dict[str, List[Run]] = {}
            for r in runs:
                m = getattr(r, "model", None) or "claude-3-5-haiku"
                model_runs.setdefault(m, []).append(r)

            model_stats = []
            for m, m_runs in model_runs.items():
                r_ids = [r.id for r in m_runs]
                m_evals = session.query(EvalResult).filter(EvalResult.run_id.in_(r_ids)).all()
                if m_evals:
                    avg_sc = float(np.mean([e.score for e in m_evals])) * 100.0
                else:
                    avg_sc = 85.0

                costs = [r.est_cost_usd or 0.0 for r in m_runs]
                avg_c = float(np.mean(costs)) if costs else 0.001
                eff_ratio = avg_sc / max(0.0001, avg_c * 100)

                model_stats.append({
                    "model": m,
                    "avg_score": round(avg_sc, 1),
                    "avg_cost_usd": round(avg_c, 5),
                    "efficiency_ratio": round(eff_ratio, 1),
                    "runs_count": len(m_runs),
                })

            model_stats.sort(key=lambda x: x["efficiency_ratio"], reverse=True)
            best_model = model_stats[0]

            findings = [
                CopilotFinding(
                    finding=f"Model `{best_model['model']}` offers the highest cost-to-quality efficiency.",
                    evidence=f"`{best_model['model']}` delivered a {best_model['avg_score']}% quality score at ${best_model['avg_cost_usd']:.5f} per run ({best_model['efficiency_ratio']} score points per cent).",
                    recommendation=f"Use `{best_model['model']}` for standard tier workloads to minimize inference costs while maintaining golden accuracy.",
                    category="Cost",
                    priority="P2 - Medium",
                )
            ]

            direct_ans = (
                f"**`{best_model['model']}` is the most cost-efficient model.** It achieved **{best_model['avg_score']}%** quality "
                f"at only **${best_model['avg_cost_usd']:.4f} per run**."
            )

            supporting_table = [
                {
                    "Model": s["model"],
                    "Quality Score": f"{s['avg_score']}%",
                    "Cost / Run": f"${s['avg_cost_usd']:.5f}",
                    "Efficiency Index": f"{s['efficiency_ratio']}",
                    "Evaluated Runs": s["runs_count"],
                }
                for s in model_stats
            ]

            return CopilotResponse(
                query=query,
                intent="which_model_is_cost_efficient",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"Most efficient: {best_model['model']} (${best_model['avg_cost_usd']:.4f}/run, score: {best_model['avg_score']}%).",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="cost_performance",
            )

        finally:
            session.close()

    @classmethod
    def common_failure_categories(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "What are the most common failure categories?",
    ) -> CopilotResponse:
        """7. Breakdown distribution across 14 failure classifications."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            if not runs or not evals:
                return CopilotResponse(
                    query=query,
                    intent="common_failure_categories",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Categorize failures
            evals_by_run = {}
            for e in evals:
                evals_by_run.setdefault(e.run_id, []).append(e)

            category_counts: Dict[str, int] = {}
            total_fails = 0

            for r in runs:
                r_evals = evals_by_run.get(r.id, [])
                failed_m = [e.metric_name for e in r_evals if not e.passed]
                if failed_m:
                    total_fails += 1
                    # Classify category
                    if any("tool_selection" in m for m in failed_m):
                        cat = FailureCategory.WRONG_TOOL_SELECTION.value
                    elif any("argument" in m for m in failed_m):
                        cat = FailureCategory.INCORRECT_TOOL_ARGUMENTS.value
                    elif any("latency" in m for m in failed_m) or (r.latency_ms or 0) > 5000:
                        cat = FailureCategory.TIMEOUT.value
                    elif any("ground" in m or "context" in m for m in failed_m):
                        cat = FailureCategory.RETRIEVAL_FAILURE.value
                    else:
                        cat = FailureCategory.WRONG_FINAL_ANSWER.value
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            if total_fails == 0:
                return CopilotResponse(
                    query=query,
                    intent="common_failure_categories",
                    direct_answer="There are zero recorded assertion failures in the active evaluation dataset.",
                    findings=[
                        CopilotFinding(
                            finding="Zero failures recorded.",
                            evidence=f"All {len(runs)} runs passed all evaluation checks.",
                            recommendation="Expand test dataset coverage.",
                            priority="P3 - Low",
                        )
                    ],
                    drill_down_section="failures",
                )

            sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            top_cat, top_cnt = sorted_cats[0]
            top_pct = round((top_cnt / max(1, total_fails)) * 100.0, 1)

            findings = [
                CopilotFinding(
                    finding=f"**{top_cat}** is the most prevalent failure classification.",
                    evidence=f"{top_cnt} of {total_fails} failing executions ({top_pct}%) fall under '{top_cat}'.",
                    recommendation="Focus engineering effort on prompt grounding and tool signature validation to eliminate this cluster.",
                    category="Reliability",
                    priority="P0 - Blocker" if top_pct > 40 else "P1 - High",
                )
            ]

            direct_ans = (
                f"The most common failure category is **{top_cat}**, accounting for **{top_cnt} of {total_fails} failures ({top_pct}%)**."
            )

            supporting_table = [
                {
                    "Failure Classification": cat,
                    "Occurrences": cnt,
                    "Percentage of Failures": f"{(cnt/max(1, total_fails)*100):.1f}%",
                }
                for cat, cnt in sorted_cats
            ]

            return CopilotResponse(
                query=query,
                intent="common_failure_categories",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"Top category: {top_cat} ({top_pct}% of failures).",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="failures",
            )

        finally:
            session.close()

    @classmethod
    def rag_poor_retrieval_queries(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which RAG queries have poor retrieval?",
    ) -> CopilotResponse:
        """8. Isolate queries with low context precision and answer faithfulness."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            rag_evals = [e for e in evals if any(k in e.metric_name for k in ["groundedness", "faithfulness", "context_precision"])]
            
            if not rag_evals:
                return CopilotResponse(
                    query=query,
                    intent="rag_poor_retrieval_queries",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Map evals to runs
            run_map = {r.id: r for r in runs}
            poor_queries = []

            for e in rag_evals:
                if not e.passed or (e.score is not None and e.score < 0.80):
                    r = run_map.get(e.run_id)
                    if r:
                        poor_queries.append({
                            "task_id": r.task_id or f"TASK-{r.id}",
                            "query": r.query or f"Run #{r.id}",
                            "metric": e.metric_name,
                            "score": round(e.score, 3) if e.score is not None else 0.0,
                            "details": e.details or "Context grounding below threshold",
                        })

            if not poor_queries:
                return CopilotResponse(
                    query=query,
                    intent="rag_poor_retrieval_queries",
                    direct_answer="All RAG and retrieval queries passed groundedness and context precision thresholds (> 80%).",
                    findings=[
                        CopilotFinding(
                            finding="High RAG grounding across all evaluated queries.",
                            evidence=f"Analyzed {len(rag_evals)} RAG assertions with zero detected grounding failures.",
                            recommendation="Continuously monitor hallucination scores as knowledge base scales.",
                            priority="P3 - Low",
                        )
                    ],
                    drill_down_section="rag_analytics",
                )

            worst_q = sorted(poor_queries, key=lambda x: x["score"])[0]

            findings = [
                CopilotFinding(
                    finding=f"Task `{worst_q['task_id']}` exhibited poor retrieval grounding ({worst_q['score']}).",
                    evidence=f"{len(poor_queries)} queries failed context grounding assertions. Query: '{worst_q['query'][:80]}...'",
                    recommendation="Increase top-K document retrieval count and re-index dense embeddings with larger chunk overlap.",
                    category="RAG & Grounding",
                    priority="P1 - High",
                )
            ]

            direct_ans = (
                f"Found **{len(poor_queries)} RAG queries** with poor retrieval or low grounding scores. "
                f"Lowest grounding was **Task `{worst_q['task_id']}`** (score: {worst_q['score']})."
            )

            supporting_table = poor_queries[:8]

            return CopilotResponse(
                query=query,
                intent="rag_poor_retrieval_queries",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{len(poor_queries)} low-grounding queries identified. Lowest: {worst_q['task_id']}.",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="rag_analytics",
            )

        finally:
            session.close()

    @classmethod
    def unnecessary_tool_calls(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "Which tools are being called unnecessarily?",
    ) -> CopilotResponse:
        """9. Detect redundant or unneeded tool calls across runs."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            unnec_evals = [e for e in evals if "unnecessary" in e.metric_name and not e.passed]
            
            tool_steps = [s for s in steps if getattr(s, "step_type", None) in ["tool", "tool_call"]]
            if not tool_steps:
                return CopilotResponse(
                    query=query,
                    intent="unnecessary_tool_calls",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            # Analyze repeated identical tool calls within same run
            runs_tool_seq: Dict[int, List[str]] = {}
            for s in tool_steps:
                runs_tool_seq.setdefault(s.run_id, []).append(getattr(s, "tool_name", None) or getattr(s, "name", None) or "unknown")

            redundant_calls_count = 0
            redundant_tools: Dict[str, int] = {}
            for r_id, seq in runs_tool_seq.items():
                seen = set()
                for t in seq:
                    if t in seen:
                        redundant_calls_count += 1
                        redundant_tools[t] = redundant_tools.get(t, 0) + 1
                    seen.add(t)

            total_unnec = len(unnec_evals) + redundant_calls_count

            if total_unnec == 0:
                return CopilotResponse(
                    query=query,
                    intent="unnecessary_tool_calls",
                    direct_answer="No unnecessary or redundant tool calls were detected in the evaluation dataset.",
                    findings=[
                        CopilotFinding(
                            finding="Tool execution sequences are clean and efficient.",
                            evidence=f"Analyzed {len(tool_steps)} tool calls across {len(runs)} runs with zero repeated or redundant invocations.",
                            recommendation="Maintain clean tool routing instructions in prompt.",
                            priority="P3 - Low",
                        )
                    ],
                    drill_down_section="tool_analytics",
                )

            top_redundant = sorted(redundant_tools.items(), key=lambda x: x[1], reverse=True)[0] if redundant_tools else ("tool_execution", total_unnec)

            findings = [
                CopilotFinding(
                    finding=f"Tool `{top_redundant[0]}` is repeatedly called within the same task execution.",
                    evidence=f"Detected {top_redundant[1]} redundant repeat invocations of `{top_redundant[0]}` across evaluated runs.",
                    recommendation="Instruct the agent explicitly in the system prompt to cache intermediate tool outputs and prevent duplicate queries.",
                    category="Tool Selection",
                    priority="P2 - Medium",
                )
            ]

            direct_ans = (
                f"Detected **{total_unnec} redundant or unnecessary tool calls**. "
                f"The tool called most redundantly is **`{top_redundant[0]}`** ({top_redundant[1]} repeated invocations)."
            )

            supporting_table = [
                {"Tool Name": t, "Redundant Invocations": cnt}
                for t, cnt in redundant_tools.items()
            ]

            return CopilotResponse(
                query=query,
                intent="unnecessary_tool_calls",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{total_unnec} redundant tool calls detected. Worst: {top_redundant[0]}.",
                recommendations=[f.recommendation for f in findings],
                supporting_table=supporting_table,
                drill_down_section="tool_analytics",
            )

        finally:
            session.close()

    @classmethod
    def what_to_investigate_first(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str = "What should I investigate first?",
    ) -> CopilotResponse:
        """10. Synthesize and rank prioritized engineering triage checklist."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            if not runs:
                return CopilotResponse(
                    query=query,
                    intent="what_to_investigate_first",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            findings: List[CopilotFinding] = []

            # 1. Critical assertion failures
            evals_by_run = {}
            for e in evals:
                evals_by_run.setdefault(e.run_id, []).append(e)

            failed_runs = [r for r in runs if any(not e.passed for e in evals_by_run.get(r.id, []))]
            if failed_runs:
                findings.append(CopilotFinding(
                    finding="High Priority: Resolve Failing Test Assertions",
                    evidence=f"{len(failed_runs)} of {len(runs)} evaluated test runs ({round(len(failed_runs)/len(runs)*100, 1)}%) failed acceptance criteria.",
                    recommendation="Open the Failures view and inspect the root-cause diagnosis for the top failing tasks.",
                    priority="P0 - Blocker",
                    category="Reliability",
                ))

            # 2. Tool Errors
            tool_steps = [s for s in steps if getattr(s, "step_type", None) in ["tool", "tool_call"]]
            failed_tool_steps = [s for s in tool_steps if getattr(s, "status", "") in ["error", "failed"] or s.error]
            if failed_tool_steps:
                findings.append(CopilotFinding(
                    finding="Medium Priority: Fix Tool Invocation Exceptions",
                    evidence=f"{len(failed_tool_steps)} tool calls encountered runtime errors during execution.",
                    recommendation="Review tool parameter schemas and error handling in tool definitions.",
                    priority="P1 - High",
                    category="Tool Selection",
                ))

            # 3. Latency outliers
            lats = [r.latency_ms or 0.0 for r in runs]
            p95_lat = float(np.percentile(lats, 95)) if lats else 0.0
            if p95_lat > 4000.0:
                findings.append(CopilotFinding(
                    finding="Medium Priority: Optimize P95 Latency Bottlenecks",
                    evidence=f"P95 latency reached {p95_lat:.0f} ms ({p95_lat/1000.0:.2f}s), approaching SLA limits.",
                    recommendation="Inspect Gantt timeline in Traces to identify slow LLM or retrieval spans.",
                    priority="P2 - Medium",
                    category="Performance",
                ))

            if not findings:
                findings.append(CopilotFinding(
                    finding="System Health is Optimal",
                    evidence="All evaluation metrics, tool executions, and latency budgets passed golden thresholds.",
                    recommendation="Continue adding edge cases and monitor production telemetry.",
                    priority="P3 - Low",
                    category="Reliability",
                ))

            direct_ans = (
                f"**Prioritized Triage Order:**\n\n"
                + "\n".join([f"{idx+1}. **[{f.priority.split()[0]}] {f.finding}** - {f.recommendation}" for idx, f in enumerate(findings)])
            )

            return CopilotResponse(
                query=query,
                intent="what_to_investigate_first",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"Identified {len(findings)} triage recommendations based on active evaluation data.",
                recommendations=[f.recommendation for f in findings],
                drill_down_section="failures",
            )

        finally:
            session.close()

    @classmethod
    def general_telemetry_diagnosis(
        cls,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
        query: str,
    ) -> CopilotResponse:
        """Fallback natural language analysis grounded in framework telemetry."""
        session = get_session()
        try:
            runs, evals, steps = cls._fetch_telemetry(session, agent_name, version, experiment_id)
            if not runs:
                return CopilotResponse(
                    query=query,
                    intent="general_diagnosis",
                    direct_answer=cls.INSUFFICIENT_DATA_MSG,
                    insufficient_data=True,
                )

            total_runs = len(runs)
            passed_runs = sum([1 for r in runs if getattr(r, "status", "") == "passed" or (getattr(r, "overall_score", 0.0) or 0.0) >= 0.85])
            pass_rate = round((passed_runs / max(1, total_runs)) * 100.0, 1)
            avg_lat = round(float(np.mean([r.latency_ms or 0.0 for r in runs])), 0) if runs else 0.0
            tot_cost = round(sum([r.est_cost_usd or 0.0 for r in runs]), 4)

            findings = [
                CopilotFinding(
                    finding=f"Evaluated {total_runs} runs with an overall pass rate of {pass_rate}%.",
                    evidence=f"{passed_runs} of {total_runs} runs passed all assertions. Mean latency: {avg_lat:.0f} ms. Total cost: ${tot_cost:.4f}.",
                    recommendation="Review individual metric breakdowns in the Metrics and Traces tabs for deeper diagnostic insights.",
                    category="Reliability",
                    priority="P2 - Medium",
                )
            ]

            direct_ans = (
                f"Based on **{total_runs} recorded evaluation runs**, your agent achieved a **{pass_rate}% pass rate** "
                f"with an average latency of **{avg_lat:.0f} ms** and total cost of **${tot_cost:.4f}**."
            )

            return CopilotResponse(
                query=query,
                intent="general_diagnosis",
                direct_answer=direct_ans,
                findings=findings,
                evidence_summary=f"{passed_runs}/{total_runs} passed ({pass_rate}%). Avg latency: {avg_lat:.0f} ms.",
                recommendations=[f.recommendation for f in findings],
                drill_down_section="overview",
            )

        finally:
            session.close()

    # ---------------------------------------------------------------------------
    # TELEMETRY FETCH HELPER
    # ---------------------------------------------------------------------------

    @classmethod
    def _fetch_telemetry(
        cls,
        session,
        agent_name: Optional[str],
        version: Optional[str],
        experiment_id: Optional[str],
    ) -> Tuple[List[Run], List[EvalResult], List[Step]]:
        """Fetch filtered runs, eval results, and steps."""
        has_filters = bool(
            (agent_name and agent_name != "All Agents") or
            (version and version != "All Versions") or
            (experiment_id and experiment_id != "All Datasets/Experiments")
        )
        q = session.query(Run)
        if experiment_id and experiment_id != "All Datasets/Experiments":
            q = q.filter(Run.experiment_id == experiment_id)
        if agent_name and agent_name != "All Agents":
            q = q.filter(Run.agent_name == agent_name)
        if version and version != "All Versions":
            q = q.filter((Run.agent_version == version) | (Run.dataset_version == version))

        runs = q.order_by(Run.id.desc()).limit(100).all()
        if not runs and not has_filters:
            # Fallback to recent runs only if no specific filter was applied
            runs = session.query(Run).order_by(Run.id.desc()).limit(50).all()

        run_ids = [r.id for r in runs] if runs else []
        evals = session.query(EvalResult).filter(EvalResult.run_id.in_(run_ids)).all() if run_ids else []
        steps = session.query(Step).filter(Step.run_id.in_(run_ids)).all() if run_ids else []

        return runs, evals, steps
