"""
Section 4: Evaluation Runs - Batch Dataset Suites, Scorecards, Run Catalog, and Failure Drill-downs.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from src.core.entities import EvaluationResult, Trace
from src.evaluation.batch_runner import BatchEvaluationRunner
from dashboard.views.common import (
    navigate_to,
    render_kpi_card,
    render_section_header,
    render_failure_analysis,
    steps_to_spans,
)


def render_eval_runs(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    scoring_config: ScoringConfig,
    load_steps_fn: Optional[Any] = None,
    tc_manager: Optional[Any] = None,
):
    render_section_header(
        title="Evaluation Runs & Batch Suites",
        subtitle="Comprehensive log of all benchmark batch suites and individual test executions with drill-downs.",
        breadcrumb="OBSERVABILITY // EVALUATION RUNS",
        action_badge="EVALUATION SUITES",
    )

    if filtered_runs.empty:
        st.info("No evaluation runs match the active filter criteria. Adjust the global filters in the sidebar.")
        return

    runner = BatchEvaluationRunner()

    # Two Top Tabs: Batch Suites vs All Individual Runs
    tab_suites, tab_individual = st.tabs([
        "📦 Batch Evaluation Suites (Datasets)",
        "📋 All Individual Test Runs Catalog",
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: Batch Evaluation Suites
    # ---------------------------------------------------------------------------
    with tab_suites:
        # Find distinct experiment IDs that have multiple runs or dataset associations
        exp_ids = filtered_runs["experiment_id"].dropna().unique().tolist()

        if not exp_ids:
            st.info("No batch evaluation suites logged yet. Run a dataset in the 'Datasets' section to view batch runs.")
        else:
            sel_exp_id = st.selectbox(
                "Select Batch Evaluation Suite",
                exp_ids,
                index=0,
                format_func=lambda eid: f"Suite #{eid} — ({len(filtered_runs[filtered_runs['experiment_id'] == eid])} tests) · {filtered_runs[filtered_runs['experiment_id'] == eid].iloc[0].get('agent_name', 'Agent')}",
                key="eval_runs_suite_selector"
            )

            # Get full evaluation run summary
            summary = runner.get_evaluation_run_summary(sel_exp_id)

            # If summary has 0 tests in batch_runner cache, compute directly from filtered_runs
            if summary.get("total_tests", 0) == 0:
                suite_runs = filtered_runs[filtered_runs["experiment_id"] == sel_exp_id]
                lats = suite_runs["latency_ms"].dropna().tolist()
                costs = suite_runs["est_cost_usd"].dropna().tolist()
                toks = suite_runs["total_tokens"].dropna().tolist()

                summary = {
                    "experiment_id": sel_exp_id,
                    "dataset_name": suite_runs.iloc[0].get("dataset_name", "Dataset Suite"),
                    "dataset_version": suite_runs.iloc[0].get("dataset_version", "1.0"),
                    "agent_name": suite_runs.iloc[0].get("agent_name", "DemoAgent"),
                    "model": suite_runs.iloc[0].get("model", "claude-3-5-haiku"),
                    "total_tests": len(suite_runs),
                    "passed": sum(1 for _, r in suite_runs.iterrows() if r.get("score", 1.0) >= 0.8),
                    "failed": sum(1 for _, r in suite_runs.iterrows() if r.get("score", 1.0) < 0.8),
                    "skipped": 0,
                    "overall_score": float(suite_runs["score"].mean() * 100) if "score" in suite_runs.columns else 90.0,
                    "metric_scores": {},
                    "average_latency_ms": float(np.mean(lats)) if lats else 0.0,
                    "p95_latency_ms": float(np.percentile(lats, 95)) if lats else 0.0,
                    "total_cost_usd": sum(costs),
                    "total_tokens": sum(toks),
                    "test_runs": [],
                }

            # -------------------------------------------------------------------
            # Full Evaluation Run Scorecard
            # -------------------------------------------------------------------
            st.markdown(
                f""" <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;"> <div> <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">BATCH SUITE SCORECARD</div> <div style="font-size: 18px; font-weight: 800; color: #f8fafc; margin-top: 2px;"> {summary.get('dataset_name', 'Dataset')} <span style="color: #64748b; font-size: 13px;">(v{summary.get('dataset_version', '1.0')} on {summary.get('agent_name', 'Agent')})</span> </div> </div> <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px;"> Model: <strong style="color: #38bdf8;">{summary.get('model', 'claude-3-5-haiku')}</strong> </span> </div> """,
                unsafe_allow_html=True,
            )

            # Scorecard Row 1: Test Counts & Overall Score
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            tot = summary.get("total_tests", 0)
            pas = summary.get("passed", 0)
            fai = summary.get("failed", 0)
            ski = summary.get("skipped", 0)
            p_pct = (pas / tot * 100) if tot else 0.0
            f_pct = (fai / tot * 100) if tot else 0.0
            ov_score = summary.get("overall_score", 0.0)

            with sc1:
                render_kpi_card("Total Tests", str(tot), subtitle="Suite Execution", icon="🔢", accent_color="#38bdf8")
            with sc2:
                render_kpi_card("Passed", f"{pas} ({p_pct:.0f}%)", health="🟢 Healthy" if p_pct >= 90 else "🟡 Degraded", icon="✅", accent_color="#10b981")
            with sc3:
                render_kpi_card("Failed", f"{fai} ({f_pct:.0f}%)", health="🔴 Critical" if fai > 0 else "🟢 Healthy", icon="❌", accent_color="#f43f5e")
            with sc4:
                render_kpi_card("Skipped", str(ski), subtitle="Disabled Tests", icon="⚪", accent_color="#64748b")
            with sc5:
                render_kpi_card("Overall Score", f"{ov_score:.1f}%", subtitle="Weighted Suite Score", health="🟢 Healthy" if ov_score >= 85 else "🟡 Degraded", icon="🎯", accent_color="#818cf8")

            # Scorecard Row 2: Latency, Cost, Tokens
            sc6, sc7, sc8, sc9 = st.columns(4)
            with sc6:
                render_kpi_card("Avg Latency", f"{summary.get('average_latency_ms', 0):.0f} ms", subtitle="Mean Duration", icon="⏱️", accent_color="#38bdf8")
            with sc7:
                render_kpi_card("P95 Latency", f"{summary.get('p95_latency_ms', 0):.0f} ms", subtitle="95th Percentile", icon="⚡", accent_color="#f59e0b")
            with sc8:
                render_kpi_card("Total Cost", f"${summary.get('total_cost_usd', 0):.4f}", subtitle="Model + Tool API", icon="🪙", accent_color="#10b981")
            with sc9:
                render_kpi_card("Total Tokens", f"{summary.get('total_tokens', 0):,}", subtitle="Input + Output Tokens", icon="🔢", accent_color="#a855f7")

            # Metric Scores Breakdown if available
            metric_scores = summary.get("metric_scores", {})
            if metric_scores:
                st.markdown("#### Per-Metric Breakdown")
                m_cols = st.columns(min(5, len(metric_scores)))
                for idx, (m_name, m_data) in enumerate(metric_scores.items()):
                    col_target = m_cols[idx % len(m_cols)]
                    with col_target:
                        m_title = m_name.replace("_", " ").title()
                        pr = m_data.get("pass_rate_pct", 0)
                        render_kpi_card(
                            m_title,
                            f"{pr:.0f}%",
                            subtitle=f"{m_data.get('passed', 0)}/{m_data.get('total', 0)} Passed",
                            health="🟢 Healthy" if pr >= 85 else "🔴 Critical",
                            accent_color="#38bdf8"
                        )

            st.markdown("<div style='margin-top: 14px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

            # -------------------------------------------------------------------
            # Individual Test Case Executions with Failure Drill-downs
            # -------------------------------------------------------------------
            st.markdown("#### Individual Test Executions in this Batch Suite")
            suite_runs_df = filtered_runs[filtered_runs["experiment_id"] == sel_exp_id]

            filter_c1, filter_c2 = st.columns([3, 1])
            with filter_c1:
                search_tc = st.text_input("Search Test Cases by ID or Query", placeholder="e.g. TC_01 or discount", key="suite_search_tc")
            with filter_c2:
                status_sub = st.selectbox("Status Filter", ["All Tests", "Failed Only", "Passed Only"], key="suite_status_sub")

            # Render individual test cards
            for _, r in suite_runs_df.iterrows():
                rid = r["run_id"]
                ev_subset = filtered_evals[filtered_evals["run_id"] == rid] if not filtered_evals.empty else pd.DataFrame()
                is_p = bool(ev_subset["passed"].all()) if not ev_subset.empty else True

                if status_sub == "Failed Only" and is_p:
                    continue
                if status_sub == "Passed Only" and not is_p:
                    continue
                if search_tc.strip():
                    q_kw = search_tc.strip().lower()
                    if q_kw not in str(r.get("task_id", "")).lower() and q_kw not in str(r.get("query", "")).lower():
                        continue

                status_pill = (
                    '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#10b981;"></span>PASSED</span>'
                    if is_p else
                    '<span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(244, 63, 94, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#f43f5e;"></span>FAILED</span>'
                )

                with st.container(border=True):
                    tc1, tc2, tc3, tc4, tc5 = st.columns([1.5, 3.5, 1.2, 1.2, 2.6])
                    with tc1:
                        st.markdown(f"<span style='font-family: \"JetBrains Mono\", monospace; font-weight: 700; color: #f8fafc;'>Run #{rid}</span><br>{status_pill}", unsafe_allow_html=True)
                    with tc2:
                        q_snip = (str(r.get('query', ''))[:55] + "...") if len(str(r.get('query', ''))) > 55 else str(r.get('query', ''))
                        st.markdown(
                            f""" <span style="color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{r.get('task_id', 'Task')}</span> <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">{q_snip}</div> """,
                            unsafe_allow_html=True,
                        )
                    with tc3:
                        st.markdown(f"⏱️ **{r.get('latency_ms', 0):.0f} ms**<br><span style='font-size: 11px; color: #64748b;'>Latency</span>", unsafe_allow_html=True)
                    with tc4:
                        st.markdown(f"🪙 **${r.get('est_cost_usd', 0):.4f}**<br><span style='font-size: 11px; color: #64748b;'>{r.get('total_tokens', 0)} tok</span>", unsafe_allow_html=True)
                    with tc5:
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("Trace ➔", key=f"suite_tr_{rid}", width="stretch", help="Inspect execution trace and spans"):
                                navigate_to("🔍 Traces", selected_run_id=rid)
                        with b2:
                            if not is_p:
                                if st.button("RCA ➔", key=f"suite_rca_{rid}", width="stretch", help="Open Root Cause Analysis for this failure"):
                                    navigate_to("🚨 Failures", selected_run_id=rid)
                            else:
                                st.caption("✅ Clean")

                    if not is_p:
                        with st.expander("🔬 Failure Analysis", expanded=True):
                            t_case = tc_manager.get_test_case(r.get("task_id")) if tc_manager else None
                            steps = load_steps_fn(int(rid)) if load_steps_fn else []
                            span_objects = steps_to_spans(steps) if steps else []
                            trace_obj = Trace(
                                task_id=r.get("task_id", ""),
                                query=r.get("query", ""),
                                final_answer=r.get("final_answer", ""),
                                spans=span_objects,
                                is_mock=bool(r.get("is_mock", True)),
                            )
                            trace_obj.latency_ms = float(r.get("latency_ms") or 0.0)
                            eval_objs = []
                            for _, erow in ev_subset.iterrows():
                                eval_objs.append(EvaluationResult(
                                    metric_name=erow["metric_name"],
                                    score=float(erow["score"]),
                                    passed=bool(erow["passed"]),
                                    threshold=float(erow.get("threshold", 1.0) or 1.0),
                                    explanation=erow.get("details", "") or erow.get("explanation", ""),
                                ))
                            render_failure_analysis(t_case, trace_obj, eval_objs, show_all=True)

    # ---------------------------------------------------------------------------
    # TAB 2: All Individual Test Runs Catalog
    # ---------------------------------------------------------------------------
    with tab_individual:
        # Compute run verdicts and scores
        run_records = []
        pass_count = 0
        fail_count = 0

        for _, r in filtered_runs.iterrows():
            rid = r["run_id"]
            run_ev = filtered_evals[filtered_evals["run_id"] == rid] if not filtered_evals.empty else pd.DataFrame()
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
            is_pass = c_summary["passed"]
            if is_pass:
                pass_count += 1
            else:
                fail_count += 1

            run_records.append({
                "run_id": rid,
                "status": "🟢 PASSED" if is_pass else "🔴 FAILED",
                "passed_bool": is_pass,
                "weighted_score": c_summary["weighted_score"],
                "agent_name": r.get("agent_name", "DemoAgent"),
                "agent_version": r.get("agent_version", "v1.0"),
                "model": r.get("model", "haiku"),
                "task_id": r.get("task_id", ""),
                "query": r.get("query", ""),
                "latency_ms": r.get("latency_ms", 0.0),
                "tokens": r.get("total_tokens", 0),
                "cost_usd": r.get("est_cost_usd", 0.0),
                "is_mock": r.get("is_mock", True),
                "integration_type": r.get("integration_type", "mock"),
                "created_at": r.get("created_at", ""),
            })

        runs_table_df = pd.DataFrame(run_records)

        # Top KPI Metrics for Runs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card("Total Runs", str(len(runs_table_df)), subtitle="Filtered Set", icon="🔢", accent_color="#38bdf8")
        with k2:
            pass_pct = (pass_count / len(runs_table_df) * 100) if runs_table_df.shape[0] > 0 else 0
            render_kpi_card("Passed Runs", f"{pass_count} ({pass_pct:.1f}%)", health="🟢 Healthy" if pass_pct >= 90 else "🟡 Degraded", icon="✅", accent_color="#10b981")
        with k3:
            fail_pct = (fail_count / len(runs_table_df) * 100) if runs_table_df.shape[0] > 0 else 0
            render_kpi_card("Failed Runs", f"{fail_count} ({fail_pct:.1f}%)", health="🟢 Healthy" if fail_pct <= 10 else "🔴 Critical", icon="⚠️", accent_color="#f43f5e")
        with k4:
            render_kpi_card("Avg Latency", f"{runs_table_df['latency_ms'].mean():.0f} ms", subtitle="Mean Runtime", icon="⏱️", accent_color="#38bdf8")
        with k5:
            render_kpi_card("Total Cost", f"${runs_table_df['cost_usd'].sum():.4f}", subtitle="Model + Tools", icon="🪙", accent_color="#10b981")

        st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # Search & status filter
        s_col1, s_col2 = st.columns([3, 1])
        with s_col1:
            search_kw = st.text_input("🔍 Search Runs by Task ID, Query, or Model", placeholder="e.g. TC_01 or weather or haiku", key="runs_search_kw")
        with s_col2:
            status_subfilter = st.selectbox("Status Filter", ["All", "Passed Only", "Failed Only"], key="runs_status_subfilter")

        filtered_catalog = runs_table_df
        if search_kw.strip():
            kw = search_kw.strip().lower()
            filtered_catalog = filtered_catalog[
                filtered_catalog["task_id"].str.lower().str.contains(kw, na=False) |
                filtered_catalog["query"].str.lower().str.contains(kw, na=False) |
                filtered_catalog["model"].str.lower().str.contains(kw, na=False)
            ]
        if status_subfilter == "Passed Only":
            filtered_catalog = filtered_catalog[filtered_catalog["passed_bool"] == True]
        elif status_subfilter == "Failed Only":
            filtered_catalog = filtered_catalog[filtered_catalog["passed_bool"] == False]

        # Pagination controls for scalable catalog rendering
        PAGE_SIZE = 15
        total_items = len(filtered_catalog)
        total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)

        p_c1, p_c2 = st.columns([3, 1])
        with p_c1:
            current_page = st.number_input(
                f"Page (1 of {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="runs_catalog_page"
            )
        with p_c2:
            st.write("")
            st.caption(f"Showing **{min(total_items, (current_page - 1) * PAGE_SIZE + 1)}–{min(total_items, current_page * PAGE_SIZE)}** of **{total_items}** runs")

        start_idx = (current_page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_items)

        st.markdown(f"### Evaluation Run Catalog ({total_items} matching)")

        # Render runs feed for current page slice
        sorted_catalog = filtered_catalog.sort_values("created_at", ascending=False)
        page_slice = sorted_catalog.iloc[start_idx:end_idx]

        for _, row in page_slice.iterrows():
            rid = row["run_id"]
            status_pill = '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#10b981;"></span>PASSED</span>' if row["passed_bool"] else '<span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(244, 63, 94, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#f43f5e;"></span>FAILED</span>'

            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns([1.3, 2.2, 2.8, 1.2, 1.2, 2.2])
                with c1:
                    st.markdown(f"<span style='font-family: \"JetBrains Mono\", monospace; font-weight: 700; color: #f8fafc;'>Run #{rid}</span><br>{status_pill}", unsafe_allow_html=True)
                with c2:
                    itype = str(row.get("integration_type", "mock")).lower()
                    if itype == "sdk":
                        ibadge = '<span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(168, 85, 247, 0.4); margin-left: 6px;">SDK</span>'
                    elif itype == "live":
                        ibadge = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.4); margin-left: 6px;">LIVE</span>'
                    else:
                        ibadge = '<span style="background: rgba(100, 116, 139, 0.2); color: #94a3b8; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(100, 116, 139, 0.4); margin-left: 6px;">MOCK</span>'
                    st.markdown(f"🤖 **{row['agent_name']}** {ibadge}<br><span style='font-size: 11px; color: #94a3b8; font-family: \"JetBrains Mono\", monospace;'>{row['agent_version']} · {row['model']}</span>", unsafe_allow_html=True)
                with c3:
                    q_text = str(row['query'])
                    q_snip = (q_text[:50] + '...') if len(q_text) > 50 else q_text
                    st.markdown(f"🧪 <span style='font-family: \"JetBrains Mono\", monospace; font-weight: 600; color: #38bdf8;'>{row['task_id']}</span><br><span style='font-size: 11.5px; color: #cbd5e1;'>{q_snip}</span>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"⏱️ **{row['latency_ms']:.0f} ms**<br><span style='font-size: 11px; color: #64748b;'>Latency</span>", unsafe_allow_html=True)
                with c5:
                    st.markdown(f"🪙 **${row['cost_usd']:.4f}**<br><span style='font-size: 11px; color: #64748b;'>Score: {row['weighted_score']:.0f}%</span>", unsafe_allow_html=True)
                with c6:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Trace ➔", key=f"btn_r_tr_{rid}", width="stretch", help="Inspect execution trace & spans"):
                            navigate_to("🔍 Traces", selected_run_id=rid)
                    with b2:
                        if not row["passed_bool"]:
                            if st.button("RCA ➔", key=f"btn_r_rca_{rid}", width="stretch", help="AI Root Cause Analysis"):
                                navigate_to("🚨 Failures", selected_run_id=rid)
                        else:
                            st.caption("✅ Clean")

                run_eval_rows = filtered_evals[filtered_evals["run_id"] == rid] if not filtered_evals.empty else pd.DataFrame()
                with st.expander(f"📊 Evaluation Breakdown — Weighted Score: {row['weighted_score']:.1f}%", expanded=False):
                    if row["is_mock"]:
                        st.warning("⚠️ **SIMULATED RUN**: Mock scores must never be presented as real evaluation results.")
                        eval_mode_badge = '<span style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(245, 158, 11, 0.4);">MOCK EVALUATION</span>'
                    else:
                        eval_mode_badge = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.4);">LIVE EVALUATION</span>'

                    st.markdown(f"**Evaluation Mode:** {eval_mode_badge}", unsafe_allow_html=True)

                    if not run_eval_rows.empty:
                        m_cols = st.columns(min(len(run_eval_rows), 5))
                        for m_idx, (_, erow) in enumerate(run_eval_rows.iterrows()):
                            m_col = m_cols[m_idx % len(m_cols)]
                            with m_col:
                                m_n = erow["metric_name"].replace("_", " ").title()
                                s_val = float(erow["score"])
                                p_val = bool(erow["passed"])
                                e_type = str(erow.get("evaluator_type", "deterministic")).lower()
                                if row["is_mock"]:
                                    t_badge = "MOCK"
                                    t_color = "#f59e0b"
                                elif e_type in ["llm_judge", "model_based", "llm_based"]:
                                    t_badge = "LLM-BASED"
                                    t_color = "#10b981"
                                elif e_type in ["semantic", "heuristic"]:
                                    t_badge = "HEURISTIC"
                                    t_color = "#c084fc"
                                else:
                                    t_badge = "DETERMINISTIC"
                                    t_color = "#38bdf8"

                                st.markdown(
                                    f""" <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 10px; margin-top: 6px;"> <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">{m_n}</div> <div style="font-size: 16px; font-weight: 800; color: {'#34d399' if p_val else '#fb7185'}; margin: 2px 0;"> {s_val:.2f} <span style="font-size: 10.5px;">({'PASS' if p_val else 'FAIL'})</span> </div> <span style="font-size: 9.5px; font-weight: 700; color: {t_color}; background: {t_color}1a; border: 1px solid {t_color}33; padding: 1px 5px; border-radius: 3px;"> [{t_badge}] </span> </div> """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("No per-metric evaluation records logged for this run.")

                if not row["passed_bool"] or (not run_eval_rows.empty and any(not erow.get("passed", True) for _, erow in run_eval_rows.iterrows())):
                    with st.expander("🔬 Failure Analysis", expanded=True):
                        t_case = tc_manager.get_test_case(row["task_id"]) if tc_manager else None
                        steps = load_steps_fn(int(rid)) if load_steps_fn else []
                        span_objects = steps_to_spans(steps) if steps else []
                        trace_obj = Trace(
                            task_id=row["task_id"],
                            query=row["query"],
                            final_answer=row.get("final_answer", ""),
                            spans=span_objects,
                            is_mock=bool(row.get("is_mock", True)),
                        )
                        trace_obj.latency_ms = float(row.get("latency_ms") or 0.0)
                        eval_objs = []
                        for _, erow in run_eval_rows.iterrows():
                            eval_objs.append(EvaluationResult(
                                metric_name=erow["metric_name"],
                                score=float(erow["score"]),
                                passed=bool(erow["passed"]),
                                threshold=float(erow.get("threshold", 1.0) or 1.0),
                                explanation=erow.get("details", "") or erow.get("explanation", ""),
                            ))
                        render_failure_analysis(t_case, trace_obj, eval_objs, show_all=True)
