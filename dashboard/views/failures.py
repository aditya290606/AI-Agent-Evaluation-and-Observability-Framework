"""
Section 7: Failures - AI-Assisted Failure & Root Cause Analysis with 3-Pillars Breakdown.
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.core.entities import EvaluationResult, Trace
from src.analysis.root_cause import RootCauseAnalyzer, FailureCategory
from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from dashboard.views.common import (
    steps_to_spans,
    navigate_to,
    render_kpi_card,
    render_section_header,
    render_failure_analysis,
)


def render_failures(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    load_steps_fn: Any,
    tc_manager: Any,
    scoring_config: ScoringConfig,
):
    render_section_header(
        title="Failure Analysis & Root Cause",
        subtitle="AI-assisted diagnosis — deterministic telemetry facts separated from inferred hypotheses.",
        breadcrumb="AGENTPULSE // FAILURES",
        action_badge="DIAGNOSTIC ENGINE",
    )

    if filtered_runs.empty:
        st.info("No runs found matching active filters.")
        return

    # Demo click-flow strip
    st.markdown(
        """ <div style="display:flex; align-items:center; gap:6px; margin-bottom:18px; background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.15); border-radius:8px; padding:10px 16px; font-size:12px; color:#64748b;"> <span style="color:#38bdf8; font-weight:700;">DEMO FLOW</span> <span style="color:#334155;">›</span> <span style="color:#f8fafc; font-weight:600;">1. Pick a failed run</span> <span style="color:#334155;">›</span> <span style="color:#f8fafc; font-weight:600;">2. Read Failure Analysis</span> <span style="color:#334155;">›</span> <span style="color:#38bdf8; font-weight:700;">3. Click "View Trace" →</span> <span style="color:#334155;">›</span> <span style="color:#f8fafc; font-weight:600;">4. Inspect Spans</span> </div> """,
        unsafe_allow_html=True,
    )

    # Pre-group evals once (eliminates N per-run pandas scans)
    eval_groups = (
        {rid: grp for rid, grp in filtered_evals.groupby("run_id")}
        if not filtered_evals.empty else {}
    )

    # Find failed runs
    failed_runs_list = []
    for _, r in filtered_runs.iterrows():
        rid = r["run_id"]
        run_ev = eval_groups.get(rid, pd.DataFrame())
        ev_objs = []
        for _, erow in run_ev.iterrows():
            ev_objs.append(EvaluationResult(
                metric_name=erow["metric_name"],
                score=float(erow["score"]),
                passed=bool(erow["passed"]),
                threshold=float(erow.get("threshold", 1.0) or 1.0),
                explanation=erow.get("details", "") or "",
            ))
        c_summary = calculate_case_scores(ev_objs, scoring_config)
        has_failed_eval = any(not e.passed for e in ev_objs)
        if not c_summary["passed"] or has_failed_eval:
            failed_runs_list.append((rid, r, c_summary, ev_objs))

    # Top KPI Cards for Failures
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        render_kpi_card("Total Failed Runs", str(len(failed_runs_list)), subtitle="Requires RCA Inspection", health="🔴 Critical" if len(failed_runs_list) > 0 else "🟢 Healthy", icon="🚨", accent_color="#f43f5e")
    with f2:
        failure_rate = (len(failed_runs_list) / len(filtered_runs) * 100) if len(filtered_runs) > 0 else 0
        render_kpi_card("Failure Rate", f"{failure_rate:.1f}%", health="🟢 Healthy" if failure_rate <= 10 else "🔴 Critical", icon="📉", accent_color="#f43f5e" if failure_rate > 10 else "#10b981")
    with f3:
        critical_violations = sum(1 for _, _, c, _ in failed_runs_list if c.get("critical_failed", False))
        render_kpi_card("Critical Gate Breaches", str(critical_violations), subtitle="Violated Critical Thresholds", health="🔴 Critical" if critical_violations > 0 else "🟢 Healthy", icon="🛡️", accent_color="#f43f5e" if critical_violations > 0 else "#10b981")
    with f4:
        avg_score_failed = (sum(c["weighted_score"] for _, _, c, _ in failed_runs_list) / len(failed_runs_list)) if failed_runs_list else 0.0
        render_kpi_card("Avg Failed Score", f"{avg_score_failed:.1f}%", subtitle="Weighted Suite Score", icon="🎯", accent_color="#f59e0b")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    if not failed_runs_list:
        st.success("🎉 **Zero Failures Detected!** All executed runs and evaluation metrics passed within budget.")
        return

    # Select Failed Run to Diagnose
    failed_run_ids = [rid for rid, _, _, _ in failed_runs_list]
    default_rid = st.session_state.get("selected_run_id", failed_run_ids[0])
    if default_rid not in failed_run_ids:
        default_rid = failed_run_ids[0]

    f_sel_c1, f_sel_c2, f_sel_c3 = st.columns([3, 1, 1])
    with f_sel_c1:
        target_run_id = st.selectbox(
            "Select Failed Run to Diagnose:",
            failed_run_ids,
            index=failed_run_ids.index(default_rid),
            format_func=lambda x: (
                f"Run #{x} — Task: {filtered_runs[filtered_runs['run_id'] == x].iloc[0]['task_id']} "
                f"({filtered_runs[filtered_runs['run_id'] == x].iloc[0]['query'][:40]}…)"
            ),
            key="rca_run_selector"
        )
    with f_sel_c2:
        st.write("")
        st.write("")
        if st.button("🔍 View Full Trace →", key="btn_rca_to_trace", type="primary"):
            navigate_to("🔍 Traces", selected_run_id=target_run_id)
    with f_sel_c3:
        st.write("")
        st.write("")
        if st.button("🚀 All Runs →", key="btn_rca_to_runs", help="Go to Evaluation Runs"):
            navigate_to("🚀 Evaluation Runs", selected_run_id=target_run_id)

    # Get data for selected run
    matched_run = next(item for item in failed_runs_list if item[0] == target_run_id)
    _, run_row, single_run_summary, case_res_objs = matched_run

    steps = load_steps_fn(int(target_run_id))
    span_objects = steps_to_spans(steps)

    run_trace = Trace(
        task_id=run_row["task_id"],
        query=run_row["query"],
        final_answer=run_row["final_answer"],
        spans=span_objects,
        is_mock=bool(run_row.get("is_mock", True)),
    )
    run_trace.latency_ms = float(run_row.get("latency_ms") or 0.0)
    run_tc = tc_manager.get_test_case(run_row["task_id"])

    # Run AI Root Cause Analysis
    rca_diagnosis = RootCauseAnalyzer.analyze(test_case=run_tc, trace=run_trace, evaluation_results=case_res_objs)

    # Classification & Inferred Confidence Banner in luxury card
    st.markdown(
        f""" <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 12px; padding: 18px 20px; margin-bottom: 20px;"> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"> <div style="font-size: 11px; font-weight: 700; color: #fb7185; text-transform: uppercase; letter-spacing: 0.08em;">DIAGNOSTIC VERDICT</div> <span style="background: rgba(244, 63, 94, 0.2); color: #fda4af; font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 3px 10px; border-radius: 9999px; font-weight: 700;"> CONFIDENCE: {rca_diagnosis.confidence * 100:.0f}% </span> </div> <div style="font-size: 20px; font-weight: 800; color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif;"> {rca_diagnosis.failure_category.value} </div> <div style="font-size: 13px; color: #cbd5e1; margin-top: 6px;"> <strong>Impact:</strong> {rca_diagnosis.impact} &nbsp;|&nbsp; <strong>Evidence:</strong> <code>{rca_diagnosis.evidence}</code> </div> </div> """,
        unsafe_allow_html=True,
    )

    st.info("⚠️ **Inferred Diagnostic Separation Notice**: This diagnosis is an automated hypothesis generated from telemetry heuristics. Always verify against the observed facts below before modifying prompts or tools.")

    # Structured Failure Analysis Card (Failure, Expected, Actual, Evidence, Likely Root Cause, Recommendation)
    render_failure_analysis(test_case=run_tc, trace=run_trace, eval_results=case_res_objs, show_all=True)

    # Expected vs Actual Diff Display
    exp_col, act_col = st.columns(2)
    with exp_col:
        st.markdown(
            f""" <div style="background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px;"> <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px;">📋 Expected Behavior</div> <div style="font-size: 13px; color: #e2e8f0;">{rca_diagnosis.expected}</div> </div> """,
            unsafe_allow_html=True,
        )
    with act_col:
        st.markdown(
            f""" <div style="background: rgba(244, 63, 94, 0.06); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px;"> <div style="font-size: 11px; font-weight: 700; color: #fb7185; text-transform: uppercase; margin-bottom: 4px;">💥 Actual Execution Result</div> <div style="font-size: 13px; color: #e2e8f0;">{rca_diagnosis.actual}</div> </div> """,
            unsafe_allow_html=True,
        )

    # 3-Pillars Tabbed Breakdown
    p_tab_facts, p_tab_diag, p_tab_remed = st.tabs([
        "🔍 Observed Facts (Deterministic Telemetry)",
        "🧠 Inferred Diagnosis (AI Hypothesis)",
        "💡 Suggested Remediation (Actionable)",
    ])

    with p_tab_facts:
        st.markdown("##### Concrete Observations from Telemetry (Zero Speculation):")
        facts = rca_diagnosis.observed_facts
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f1.metric("Tools Called", len(facts.tools_called))
        col_f2.metric("Span Errors", len(facts.errors_detected))
        col_f3.metric("Latency", f"{facts.latency_ms:.0f} ms", delta=f"{facts.latency_ms - facts.latency_budget_ms:+.0f} ms budget")
        col_f4.metric("Retrieval Docs", facts.retrieved_documents_count)

        if facts.tools_called:
            st.markdown("**Tools Executed:**")
            st.dataframe(pd.DataFrame(facts.tools_called)[["span_ref", "tool_name", "input", "status"]], width="stretch")

        if facts.errors_detected:
            st.markdown("**Span Errors Logged:**")
            for err in facts.errors_detected:
                st.error(f"**{err['span_ref']}**: {err['error']}")

        if facts.failed_metric_details:
            st.markdown("**Failed Evaluation Checks:**")
            for fmd in facts.failed_metric_details:
                st.markdown(f"- {fmd}")

    with p_tab_diag:
        st.markdown("##### Inferred Diagnostic Hypothesis:")
        st.markdown(f"- **Primary Category:** `{rca_diagnosis.failure_category.value}`")
        st.markdown(f"- **Diagnostic Confidence:** `{rca_diagnosis.confidence:.2f}` (calibrated algorithmic inference)")
        if rca_diagnosis.inferred_diagnosis.get("root_cause"):
            st.markdown(f"- **Inferred Mechanism:** {rca_diagnosis.inferred_diagnosis['root_cause']}")
        if facts.retry_loops_detected > 0:
            st.warning(f"Detected {facts.retry_loops_detected} redundant loop retries during execution.")

    with p_tab_remed:
        st.markdown("##### Recommended Actionable Remediations:")
        st.success(f"**Immediate Action:** {rca_diagnosis.suggested_remediation}")
        if rca_diagnosis.recommendations:
            st.markdown("**Engineering Checklist:**")
            for rec in rca_diagnosis.recommendations:
                st.markdown(f"- {rec}")
