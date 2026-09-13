"""
Section 8: Experiments / Versions - Version Comparison, Multi-Metric Quality Gates, and Regression Transition Matrix.

Includes:
 - Tab 1: Agent Version Comparison — focused v1 vs v2 comparison with 80% quality gate
 - Tab 2: Advanced Experiment Comparison — original full experiment/version selector
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.regression import (
    VersionComparator,
    QualityGatePolicy,
    Direction,
    TransitionType,
    VersionComparisonReport,
)

from dashboard.views.common import render_section_header, apply_plotly_theme

# Metrics surfaced in the dedicated agent version comparison
AGENT_VERSION_METRICS = [
    "overall_score",
    "task_success",
    "tool_accuracy",
    "answer_quality",
    "groundedness",
    "latency",
]

METRIC_LABELS = {
    "overall_score": "Overall Score",
    "task_success": "Task Success",
    "tool_accuracy": "Tool Accuracy",
    "answer_quality": "Answer Correctness",
    "groundedness": "Groundedness",
    "latency": "Latency",
}

QUALITY_GATE_THRESHOLD = 80.0  # Hard floor — any version below this is a gate fail


def _delta_badge(delta: float, unit: str = "%", direction: Direction = Direction.HIGHER_IS_BETTER) -> str:
    """Return an HTML badge string for a delta value."""
    good = (delta > 0) if direction == Direction.HIGHER_IS_BETTER else (delta < 0)
    bad = (delta < 0) if direction == Direction.HIGHER_IS_BETTER else (delta > 0)
    if good:
        color, arrow = "#10b981", "▲"
        bg = "rgba(16,185,129,0.12)"
        border = "rgba(16,185,129,0.3)"
    elif bad:
        color, arrow = "#f43f5e", "▼"
        bg = "rgba(244,63,94,0.12)"
        border = "rgba(244,63,94,0.3)"
    else:
        color, arrow = "#94a3b8", "●"
        bg = "rgba(148,163,184,0.10)"
        border = "rgba(148,163,184,0.2)"

    prefix = "+" if delta > 0 else ""
    val_str = f"{prefix}{delta:.1f} {unit}" if unit != "ms" else f"{prefix}{delta:.0f} ms"
    return (
        f'<span style="background:{bg}; color:{color}; border:1px solid {border}; '
        f'font-size:11.5px; font-weight:700; padding:2px 8px; border-radius:9999px; '
        f'font-family:\'JetBrains Mono\',monospace; display:inline-block;">'
        f'{arrow} {val_str}</span>'
    )


def _pct_badge(pct: float) -> str:
    if pct > 0:
        color, bg, brd = "#10b981", "rgba(16,185,129,0.10)", "rgba(16,185,129,0.25)"
        txt = f"+{pct:.1f}%"
    elif pct < 0:
        color, bg, brd = "#f43f5e", "rgba(244,63,94,0.10)", "rgba(244,63,94,0.25)"
        txt = f"{pct:.1f}%"
    else:
        color, bg, brd = "#64748b", "transparent", "transparent"
        txt = "0%"
    return (
        f'<span style="background:{bg}; color:{color}; border:1px solid {brd}; '
        f'font-size:11px; font-weight:700; padding:2px 7px; border-radius:9999px;">'
        f'{txt}</span>'
    )


def _render_agent_version_comparison(all_runs: pd.DataFrame, all_evals: pd.DataFrame):
    """Dedicated, focused agent version comparison tab."""

    st.markdown(
        """ <div style="margin-bottom:20px;"> <div style="font-size:12px; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">AGENT VERSION COMPARISON</div> <div style="font-size:15px; color:#94a3b8;"> Select an agent and two versions to compare head-to-head across the same evaluation dataset. Uses the <strong style="color:#f8fafc;">80% quality gate</strong> and checks all 6 key metrics independently. </div> </div> """,
        unsafe_allow_html=True,
    )

    # ── Agent selector ──────────────────────────────────────────────────────
    all_agents = sorted(all_runs["agent_name"].dropna().unique().tolist()) if not all_runs.empty else []
    if not all_agents:
        st.info("No evaluation runs found. Run evaluations first to enable version comparison.")
        return

    sel_agent = st.selectbox("🤖 Select Agent", all_agents, key="vc_agent")

    agent_runs = all_runs[all_runs["agent_name"] == sel_agent]
    agent_evals = all_evals[all_evals["run_id"].isin(agent_runs["run_id"])] if not all_evals.empty else pd.DataFrame()

    available_versions = sorted(agent_runs["agent_version"].dropna().unique().tolist())
    if len(available_versions) < 1:
        st.warning(f"No version data found for **{sel_agent}**.")
        return

    # ── Version selectors ───────────────────────────────────────────────────
    vc1, vc2, vc3 = st.columns([1, 1, 1])
    with vc1:
        v1 = st.selectbox("🅰 Baseline Version (v1)", available_versions, index=0, key="vc_v1")
    with vc2:
        default_v2_idx = min(1, len(available_versions) - 1)
        v2 = st.selectbox("🅱 Target Version (v2)", available_versions, index=default_v2_idx, key="vc_v2")
    with vc3:
        # Dataset / experiment filter
        all_experiments = ["All Datasets"] + sorted(
            [e for e in agent_runs["experiment_id"].dropna().unique().tolist() if e != "legacy"]
        )
        sel_dataset = st.selectbox("📁 Dataset / Experiment", all_experiments, index=0, key="vc_dataset")

    if v1 == v2:
        st.warning("Baseline and target versions are the same. Select two different versions.")
        return

    # ── Slice runs ──────────────────────────────────────────────────────────
    def _filter(ver):
        r = agent_runs[agent_runs["agent_version"] == ver]
        if sel_dataset != "All Datasets":
            r = r[r["experiment_id"] == sel_dataset]
        return r

    b_runs = _filter(v1)
    t_runs = _filter(v2)
    b_evals = agent_evals[agent_evals["run_id"].isin(b_runs["run_id"])] if not agent_evals.empty else pd.DataFrame()
    t_evals = agent_evals[agent_evals["run_id"].isin(t_runs["run_id"])] if not agent_evals.empty else pd.DataFrame()

    if b_runs.empty or t_runs.empty:
        st.warning(
            f"Insufficient data: **{v1}** has {len(b_runs)} runs, **{v2}** has {len(t_runs)} runs. "
            "Ensure both versions have been evaluated."
        )
        return

    # ── Run comparison ──────────────────────────────────────────────────────
    policy = QualityGatePolicy(
        max_task_success_drop_pct=0.0,
        max_overall_score_drop_pct=0.0,
        max_tool_accuracy_drop_pct=0.0,
        max_groundedness_drop_pct=2.0,
        max_latency_increase_pct=15.0,
        max_error_rate_increase_pct=0.0,
        strict_critical_metrics=True,
    )

    report = VersionComparator.compare(
        baseline_runs=b_runs, target_runs=t_runs,
        baseline_evals=b_evals, target_evals=t_evals,
        baseline_label=f"{sel_agent} {v1}",
        target_label=f"{sel_agent} {v2}",
        policy=policy,
    )

    st.divider()

    # ── Run metadata strip ──────────────────────────────────────────────────
    meta_c1, meta_c2 = st.columns(2)
    with meta_c1:
        st.markdown(
            f""" <div style="background:rgba(56,189,248,0.07); border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:12px 16px; margin-bottom:12px;"> <div style="font-size:11px; color:#38bdf8; font-weight:700; margin-bottom:4px;"> 🅰 BASELINE — {v1}</div> <div style="font-size:13px; color:#f8fafc; font-weight:600;">{sel_agent}</div> <div style="font-size:12px; color:#94a3b8;">{len(b_runs)} runs · Model: {b_runs.iloc[0].get('model','N/A')} · Dataset: {sel_dataset}</div> </div> """,
            unsafe_allow_html=True,
        )
    with meta_c2:
        st.markdown(
            f""" <div style="background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.2); border-radius:8px; padding:12px 16px; margin-bottom:12px;"> <div style="font-size:11px; color:#10b981; font-weight:700; margin-bottom:4px;"> 🅱 TARGET — {v2}</div> <div style="font-size:13px; color:#f8fafc; font-weight:600;">{sel_agent}</div> <div style="font-size:12px; color:#94a3b8;">{len(t_runs)} runs · Model: {t_runs.iloc[0].get('model','N/A')} · Dataset: {sel_dataset}</div> </div> """,
            unsafe_allow_html=True,
        )

    # ── Quality Gate banner ─────────────────────────────────────────────────
    overall_m = report.metrics.get("overall_score")
    v1_overall = overall_m.baseline_value if overall_m else 0.0
    v2_overall = overall_m.target_value if overall_m else 0.0
    absolute_diff = v2_overall - v1_overall

    # Hard floor check: either version below 80% is a gate fail regardless
    below_floor = v2_overall < QUALITY_GATE_THRESHOLD
    gate_passed = report.quality_gate_passed and not below_floor

    if gate_passed:
        st.markdown(
            f""" <div style="background:rgba(16,185,129,0.10); border:2px solid #10b981; border-radius:12px; padding:20px 24px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;"> <div> <div style="font-size:11px; color:#10b981; font-weight:700; letter-spacing:.1em;"> ✅ QUALITY GATE</div> <div style="font-size:22px; font-weight:800; color:#34d399; margin-top:2px;"> PASSED</div> <div style="font-size:13px; color:#94a3b8; margin-top:4px;"> {v1} → {v2}: All critical metrics maintained or improved. {'No newly failing tests.' if not report.newly_failing_tests else ''} </div> </div> <div style="text-align:right;"> <div style="font-size:11px; color:#64748b; margin-bottom:2px;">OVERALL IMPROVEMENT</div> <div style="font-size:28px; font-weight:800; color:{'#10b981' if absolute_diff>=0 else '#f43f5e'};"> {'+'if absolute_diff>=0 else ''}{absolute_diff:.1f} pts </div> <div style="font-size:12px; color:#94a3b8;"> {v1_overall:.1f} → {v2_overall:.1f} (80% gate: ✅) </div> </div> </div> """,
            unsafe_allow_html=True,
        )
    else:
        reasons_html = "".join([f"<li>{r}</li>" for r in report.quality_gate_reasons])
        if below_floor:
            reasons_html += f"<li>Target version ({v2}) overall score {v2_overall:.1f}% is below the 80% quality gate floor.</li>"
        st.markdown(
            f""" <div style="background:rgba(239,68,68,0.10); border:2px solid #ef4444; border-radius:12px; padding:20px 24px; margin-bottom:20px;"> <div style="display:flex; justify-content:space-between; align-items:flex-start;"> <div> <div style="font-size:11px; color:#ef4444; font-weight:700; letter-spacing:.1em;"> 🔴 QUALITY GATE</div> <div style="font-size:22px; font-weight:800; color:#f87171; margin-top:2px;"> FAILED</div> <div style="font-size:13px; color:#94a3b8; margin-top:4px;"> Critical regressions or threshold violations detected: </div> <ul style="color:#fca5a5; font-size:13px; margin:8px 0 0 0; padding-left:20px;"> {reasons_html} </ul> </div> <div style="text-align:right; min-width:160px;"> <div style="font-size:11px; color:#64748b; margin-bottom:2px;">OVERALL DELTA</div> <div style="font-size:28px; font-weight:800; color:{'#10b981' if absolute_diff>=0 else '#f43f5e'};"> {'+'if absolute_diff>=0 else ''}{absolute_diff:.1f} pts </div> <div style="font-size:12px; color:#94a3b8;"> {v1_overall:.1f} → {v2_overall:.1f} {'(below 80% gate ❌)' if below_floor else '(80% gate ✅)'} </div> </div> </div> </div> """,
            unsafe_allow_html=True,
        )

    # ── 6-metric scorecards ─────────────────────────────────────────────────
    st.markdown("#### 📊 Metric-by-Metric Breakdown")
    metric_cols = st.columns(len(AGENT_VERSION_METRICS))

    for col, m_key in zip(metric_cols, AGENT_VERSION_METRICS):
        comp = report.metrics.get(m_key)
        label = METRIC_LABELS.get(m_key, m_key)
        if comp is None:
            with col:
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.07);'
                    f'border-radius:10px;padding:14px 12px;text-align:center;">'
                    f'<div style="font-size:11px;color:#64748b;">{label}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:#475569;">—</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            continue

        is_lat = comp.unit == "ms"
        v1_display = f"{comp.baseline_value:.0f} ms" if is_lat else f"{comp.baseline_value:.1f}%"
        v2_display = f"{comp.target_value:.0f} ms" if is_lat else f"{comp.target_value:.1f}%"
        delta_display = f"{comp.delta:+.0f} ms" if is_lat else f"{comp.delta:+.1f} pp"
        pct_display = f"{comp.pct_change:+.1f}%"

        if comp.is_improvement:
            border_color, status_icon, status_label, status_color = "#10b981", "🟢", "Improvement", "#10b981"
        elif comp.is_regression:
            border_color, status_icon, status_label, status_color = "#f43f5e", "🔴", "Regression", "#f43f5e"
        else:
            border_color, status_icon, status_label, status_color = "rgba(255,255,255,0.08)", "🟡", "Neutral", "#94a3b8"

        with col:
            st.markdown(
                f""" <div style="background:rgba(15,23,42,0.65); border:1px solid {border_color}; border-radius:10px; padding:14px 12px; text-align:center; height:100%;"> <div style="font-size:10.5px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;"> {label}</div> <div style="font-size:11px; color:#64748b; margin-bottom:2px;">v1 ({v1})</div> <div style="font-size:20px; font-weight:800; color:#f8fafc; font-family:'JetBrains Mono',monospace;">{v1_display}</div> <div style="font-size:11px; color:#64748b; margin-top:8px; margin-bottom:2px;">v2 ({v2})</div> <div style="font-size:20px; font-weight:800; color:{status_color}; font-family:'JetBrains Mono',monospace;">{v2_display}</div> <div style="margin-top:10px; margin-bottom:4px;"> <span style="font-size:12px; font-weight:700; color:{status_color}; font-family:'JetBrains Mono',monospace;">{delta_display}</span> <span style="font-size:10px; color:#64748b; margin-left:4px;">abs</span> </div> <div style="font-size:11px; color:#64748b;">{pct_display} change</div> <div style="margin-top:8px; font-size:11px; font-weight:700; color:{status_color};"> {status_icon} {status_label}</div> </div> """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Visual bar comparison ───────────────────────────────────────────────
    pct_metrics = [m for m in AGENT_VERSION_METRICS if report.metrics.get(m) and report.metrics[m].unit != "ms"]
    chart_data = []
    for m_key in pct_metrics:
        comp = report.metrics[m_key]
        chart_data.append({"Metric": METRIC_LABELS[m_key], "Version": f"{v1} (Baseline)", "Score (%)": comp.baseline_value})
        chart_data.append({"Metric": METRIC_LABELS[m_key], "Version": f"{v2} (Target)", "Score (%)": comp.target_value})

    if chart_data:
        with st.container(border=True):
            st.markdown(
                f""" <div style="font-size:14.5px; font-weight:700; color:#f8fafc; margin-bottom:4px;"> 📈 {v1} vs {v2} — Metric Radar</div> <div style="font-size:12px; color:#94a3b8; margin-bottom:10px;"> Side-by-side comparison across all quality dimensions. 80% gate floor shown as reference line.</div> """,
                unsafe_allow_html=True,
            )
            chart_df = pd.DataFrame(chart_data)
            fig = px.bar(
                chart_df, x="Metric", y="Score (%)", color="Version", barmode="group",
                range_y=[0, 105],
                color_discrete_map={
                    f"{v1} (Baseline)": "#38bdf8",
                    f"{v2} (Target)": "#10b981",
                },
            )
            # 80% gate floor reference line
            fig.add_hline(
                y=QUALITY_GATE_THRESHOLD,
                line_dash="dash", line_color="#f59e0b", line_width=1.5,
                annotation_text="80% Quality Gate",
                annotation_position="top right",
                annotation_font_color="#f59e0b",
            )
            fig.update_traces(marker_line_width=0, opacity=0.9)
            apply_plotly_theme(fig, height=320)
            st.plotly_chart(fig, width="stretch")

    # ── Latency comparison mini chart ───────────────────────────────────────
    lat_comp = report.metrics.get("latency")
    if lat_comp and (lat_comp.baseline_value > 0 or lat_comp.target_value > 0):
        with st.container(border=True):
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(f"**⏱ Latency Comparison**")
                lat_df = pd.DataFrame({
                    "Version": [f"{v1} (Baseline)", f"{v2} (Target)"],
                    "Avg Latency (ms)": [lat_comp.baseline_value, lat_comp.target_value],
                })
                fig_lat = px.bar(
                    lat_df, x="Version", y="Avg Latency (ms)",
                    color="Version",
                    color_discrete_map={
                        f"{v1} (Baseline)": "#38bdf8",
                        f"{v2} (Target)": "#10b981" if lat_comp.is_improvement else "#f43f5e",
                    },
                )
                fig_lat.update_traces(marker_line_width=0, opacity=0.9)
                apply_plotly_theme(fig_lat, height=200)
                st.plotly_chart(fig_lat, width="stretch")
            with lc2:
                lat_dir = "faster ✅" if lat_comp.is_improvement else ("slower ⚠️" if lat_comp.is_regression else "unchanged 🟡")
                st.markdown(
                    f""" <div style="padding:12px 0;"> <div style="font-size:12px; color:#64748b; margin-bottom:4px;">Baseline ({v1})</div> <div style="font-size:22px; font-weight:800; color:#f8fafc; font-family:'JetBrains Mono',monospace;"> {lat_comp.baseline_value:.0f} ms</div> <div style="font-size:12px; color:#64748b; margin-top:14px; margin-bottom:4px;">Target ({v2})</div> <div style="font-size:22px; font-weight:800; color:#f8fafc; font-family:'JetBrains Mono',monospace;"> {lat_comp.target_value:.0f} ms</div> <div style="margin-top:14px; font-size:14px; color:#94a3b8;"> {v2} is <strong>{lat_dir}</strong> ({lat_comp.delta:+.0f} ms / {lat_comp.pct_change:+.1f}%)</div> </div> """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Test transition tabs ────────────────────────────────────────────────
    st.markdown("#### 🧪 Test Case Transitions")

    nf = len(report.newly_failing_tests)
    np_ = len(report.newly_passing_tests)
    pf = len(report.persistent_failing_tests)
    pp = len(report.persistent_passing_tests)

    diff_tab1, diff_tab2, diff_tab3, diff_tab4 = st.tabs([
        f"🚨 Newly Failed ({nf})",
        f"🌟 Newly Passed ({np_})",
        f"❌ Persistent Failures ({pf})",
        f"✅ Persistent Passes ({pp})",
    ])

    with diff_tab1:
        if report.newly_failing_tests:
            st.error(
                f"⚠️ **{nf} test case{'s' if nf != 1 else ''} that passed in {v1} "
                f"now FAIL in {v2}** — these are regressions."
            )
            nf_df = pd.DataFrame([t.to_dict() for t in report.newly_failing_tests])
            display_cols = [c for c in ["task_id", "query", "baseline_score", "target_score", "target_error", "details"] if c in nf_df.columns]
            st.dataframe(nf_df[display_cols], width="stretch")
        else:
            st.success(f"🎉 **Zero newly failing tests!** {v2} maintains all tests that passed in {v1}.")

    with diff_tab2:
        if report.newly_passing_tests:
            st.success(
                f"🌟 **{np_} test case{'s' if np_ != 1 else ''} that failed in {v1} "
                f"now PASS in {v2}** — these are improvements."
            )
            np_df = pd.DataFrame([t.to_dict() for t in report.newly_passing_tests])
            display_cols = [c for c in ["task_id", "query", "baseline_score", "target_score", "baseline_error", "details"] if c in np_df.columns]
            st.dataframe(np_df[display_cols], width="stretch")
        else:
            st.info(f"No newly passing tests. Tests that failed in {v1} still fail in {v2}.")

    with diff_tab3:
        if report.persistent_failing_tests:
            st.warning(f"**{pf} test cases failing in both {v1} and {v2}:**")
            pf_df = pd.DataFrame([t.to_dict() for t in report.persistent_failing_tests])
            display_cols = [c for c in ["task_id", "query", "baseline_score", "target_score", "target_error"] if c in pf_df.columns]
            st.dataframe(pf_df[display_cols], width="stretch")
        else:
            st.success("No persistent failures — no test that failed in both versions.")

    with diff_tab4:
        st.caption(f"**{pp} test cases consistently passed across both {v1} and {v2}.**")
        if report.persistent_passing_tests:
            pp_df = pd.DataFrame([t.to_dict() for t in report.persistent_passing_tests])
            display_cols = [c for c in ["task_id", "query", "baseline_score", "target_score"] if c in pp_df.columns]
            st.dataframe(pp_df[display_cols], width="stretch")

    # ── Full metric summary table ───────────────────────────────────────────
    with st.expander("📋 Full Comparison Table (All 9 Metrics)", expanded=False):
        table_rows = []
        for m_key, comp in report.metrics.items():
            unit = comp.unit
            b_disp = f"${comp.baseline_value:.4f}" if unit == "$" else (
                f"{comp.baseline_value:.0f} ms" if unit == "ms" else f"{comp.baseline_value:.1f}%"
            )
            t_disp = f"${comp.target_value:.4f}" if unit == "$" else (
                f"{comp.target_value:.0f} ms" if unit == "ms" else f"{comp.target_value:.1f}%"
            )
            delta_disp = f"{comp.delta:+.1f} pp" if unit == "%" else (
                f"{comp.delta:+.4f}" if unit == "$" else f"{comp.delta:+.0f} {unit}"
            )
            pct_disp = f"{comp.pct_change:+.1f}%" if comp.baseline_value != 0 else "—"
            status_badge = "🟢 Improvement" if comp.is_improvement else ("🔴 Regression" if comp.is_regression else "🟡 Neutral")
            table_rows.append({
                "Metric": comp.display_name,
                f"v1 ({v1})": b_disp,
                f"v2 ({v2})": t_disp,
                "Absolute Δ": delta_disp,
                "% Change": pct_disp,
                "Status": status_badge,
                "Note": comp.explanation,
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch")


def render_experiments(all_runs: pd.DataFrame, all_evals: pd.DataFrame):
    render_section_header(
        title="Experiments & Version Regression",
        subtitle="Compare agent versions and experiments with automated regression detection, multi-metric quality gates, and test case diffing.",
        breadcrumb="ANALYSIS // EXPERIMENTS",
        action_badge="REGRESSION SUITE",
    )

    if all_runs.empty:
        st.info("No evaluation runs logged yet to compare. Run evaluations from the **📋 Reports** or **🤖 Agents** sections first.")
        return

    tab_version, tab_advanced = st.tabs([
        "🔀 Agent Version Comparison",
        "⚙️ Advanced Experiment Comparison",
    ])

    with tab_version:
        _render_agent_version_comparison(all_runs, all_evals)

    with tab_advanced:
        _render_advanced_comparison(all_runs, all_evals)


def _render_advanced_comparison(all_runs: pd.DataFrame, all_evals: pd.DataFrame):
    """Original full experiment/version comparison (preserved as-is)."""

    exp_list = sorted([e for e in all_runs["experiment_id"].dropna().unique().tolist() if e != "legacy"])
    agent_vers = sorted(
        all_runs[["agent_name", "agent_version"]].drop_duplicates()
        .apply(lambda r: f"{r['agent_name']} ({r['agent_version']})", axis=1).tolist()
    )

    cmp_c1, cmp_c2 = st.columns(2)

    with cmp_c1:
        with st.container(border=True):
            st.markdown("### 🅰️ Baseline Version / Experiment")
            b_choice = st.segmented_control("Baseline Source", ["By Experiment", "By Agent Version"], default="By Experiment", key="base_src")
            baseline_mode = b_choice if b_choice is not None else "By Experiment"
            if baseline_mode == "By Experiment" and exp_list:
                sel_base_exp = st.selectbox("Select Baseline Experiment", exp_list, index=0, key="base_exp")
                b_runs = all_runs[all_runs["experiment_id"] == sel_base_exp]
                b_label = f"Experiment: {sel_base_exp}"
            else:
                sel_base_ver = st.selectbox("Select Baseline Version", agent_vers, index=0, key="base_ver")
                b_agent_name, b_ver = sel_base_ver.split(" (")[0], sel_base_ver.split(" (")[1].rstrip(")")
                b_runs = all_runs[(all_runs["agent_name"] == b_agent_name) & (all_runs["agent_version"] == b_ver)]
                b_label = sel_base_ver

            b_evals = all_evals[all_evals["run_id"].isin(b_runs["run_id"])] if not all_evals.empty else pd.DataFrame()
            if not b_runs.empty:
                st.caption(f"**Runs:** {len(b_runs)} | **Model:** `{b_runs.iloc[0].get('model', 'N/A')}` | **Prompt:** `{b_runs.iloc[0].get('prompt_version', 'v1.0')}` | **Env:** `{b_runs.iloc[0].get('environment', 'production')}`")

    with cmp_c2:
        with st.container(border=True):
            st.markdown("### 🅱️ Target Version / Experiment")
            t_choice = st.segmented_control("Target Source", ["By Experiment", "By Agent Version"], default="By Experiment", key="targ_src")
            target_mode = t_choice if t_choice is not None else "By Experiment"
            if target_mode == "By Experiment" and exp_list:
                def_idx = 1 if len(exp_list) > 1 else 0
                sel_targ_exp = st.selectbox("Select Target Experiment", exp_list, index=def_idx, key="targ_exp")
                t_runs = all_runs[all_runs["experiment_id"] == sel_targ_exp]
                t_label = f"Experiment: {sel_targ_exp}"
            else:
                def_ver_idx = 1 if len(agent_vers) > 1 else 0
                sel_targ_ver = st.selectbox("Select Target Version", agent_vers, index=def_ver_idx, key="targ_ver")
                t_agent_name, t_ver = sel_targ_ver.split(" (")[0], sel_targ_ver.split(" (")[1].rstrip(")")
                t_runs = all_runs[(all_runs["agent_name"] == t_agent_name) & (all_runs["agent_version"] == t_ver)]
                t_label = sel_targ_ver

            t_evals = all_evals[all_evals["run_id"].isin(t_runs["run_id"])] if not all_evals.empty else pd.DataFrame()
            if not t_runs.empty:
                st.caption(f"**Runs:** {len(t_runs)} | **Model:** `{t_runs.iloc[0].get('model', 'N/A')}` | **Prompt:** `{t_runs.iloc[0].get('prompt_version', 'v1.0')}` | **Env:** `{t_runs.iloc[0].get('environment', 'production')}`")

    with st.expander("⚙️ Quality Gate Policy Configuration", expanded=False):
        pol_c1, pol_c2, pol_c3 = st.columns(3)
        with pol_c1:
            strict_crit = st.checkbox("Strict Multi-Metric Enforcement", value=True)
            max_ts_drop = st.number_input("Max Allowed Task Success Drop (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        with pol_c2:
            max_tool_drop = st.number_input("Max Allowed Tool Accuracy Drop (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            max_ground_drop = st.number_input("Max Allowed Groundedness Drop (%)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
        with pol_c3:
            max_lat_inc = st.number_input("Max Allowed Latency Degradation (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            max_err_inc = st.number_input("Max Allowed Error Rate Increase (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

        policy = QualityGatePolicy(
            max_task_success_drop_pct=max_ts_drop,
            max_overall_score_drop_pct=0.0,
            max_tool_accuracy_drop_pct=max_tool_drop,
            max_groundedness_drop_pct=max_ground_drop,
            max_latency_increase_pct=max_lat_inc,
            max_error_rate_increase_pct=max_err_inc,
            strict_critical_metrics=strict_crit,
        )

    if b_runs.empty or t_runs.empty:
        st.warning("Insufficient data in selected Baseline or Target. Select valid runs to compare.")
        return

    report = VersionComparator.compare(
        baseline_runs=b_runs, target_runs=t_runs,
        baseline_evals=b_evals, target_evals=t_evals,
        baseline_label=b_label, target_label=t_label,
        policy=policy,
    )

    st.divider()

    if report.quality_gate_passed:
        st.markdown(
            f""" <div style="background: rgba(16, 185, 129, 0.12); border: 2px solid #10b981; border-radius: 10px; padding: 18px; margin-bottom: 16px;"> <h3 style="color: #10b981; margin: 0 0 8px 0;">🟢 {report.quality_gate_status}</h3> <p style="margin: 0; color: #cbd5e1; font-size: 14px;">All critical quality thresholds maintained or improved from <strong>{report.baseline_label}</strong> to <strong>{report.target_label}</strong>. Zero regression blockers detected.</p> </div> """,
            unsafe_allow_html=True,
        )
    else:
        reasons_html = "".join([f"<li>{r}</li>" for r in report.quality_gate_reasons])
        st.markdown(
            f""" <div style="background: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 10px; padding: 18px; margin-bottom: 16px;"> <h3 style="color: #ef4444; margin: 0 0 8px 0;">🔴 {report.quality_gate_status}</h3> <p style="margin: 0 0 8px 0; color: #cbd5e1; font-size: 14px;">Critical regressions detected between <strong>{report.baseline_label}</strong> and <strong>{report.target_label}</strong>:</p> <ul style="margin: 0; color: #fca5a5; font-size: 13px;">{reasons_html}</ul> </div> """,
            unsafe_allow_html=True,
        )

    if report.improvement_highlights:
        impr_str = " &nbsp;|&nbsp; ".join([f"🚀 **{h}**" for h in report.improvement_highlights])
        st.info(f"✨ **Performance Improvements:** {impr_str}")

    st.markdown("#### 📊 Side-by-Side Multi-Metric Comparison")
    table_rows = []
    for m_key, comp in report.metrics.items():
        unit = comp.unit
        b_disp = f"${comp.baseline_value:.4f}" if unit == "$" else f"{comp.baseline_value:.1f}{unit}"
        t_disp = f"${comp.target_value:.4f}" if unit == "$" else f"{comp.target_value:.1f}{unit}"
        delta_disp = f"{comp.delta:+.1f} pp" if unit == "%" else (f"{comp.delta:+.4f}" if unit == "$" else f"{comp.delta:+.1f} {unit}")
        pct_disp = f"{comp.pct_change:+.1f}%" if comp.baseline_value != 0 else "—"
        status_badge = "🟢 Improvement" if comp.is_improvement else ("🔴 Regression" if comp.is_regression else "🟡 Neutral")
        table_rows.append({
            "Metric": comp.display_name,
            f"Baseline ({report.baseline_label[:20]})": b_disp,
            f"Target ({report.target_label[:20]})": t_disp,
            "Delta": delta_disp,
            "% Change": pct_disp,
            "Status": status_badge,
            "Assessment": comp.explanation,
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch")

    plot_metrics = ["overall_score", "task_success", "tool_accuracy", "groundedness", "answer_quality"]
    chart_data = []
    for pm in plot_metrics:
        if pm in report.metrics:
            m = report.metrics[pm]
            chart_data.append({"Metric": m.display_name, "Version": report.baseline_label, "Value (%)": m.baseline_value})
            chart_data.append({"Metric": m.display_name, "Version": report.target_label, "Value (%)": m.target_value})

    if chart_data:
        with st.container(border=True):
            st.markdown("**📈 Visual Metric Comparison**")
            chart_df = pd.DataFrame(chart_data)
            fig_comp = px.bar(
                chart_df, x="Metric", y="Value (%)", color="Version",
                barmode="group", range_y=[0, 100],
                color_discrete_sequence=["#38bdf8", "#10b981"],
            )
            fig_comp.update_traces(marker_line_width=0, opacity=0.9)
            apply_plotly_theme(fig_comp, height=310)
            st.plotly_chart(fig_comp, width="stretch")

    st.markdown("#### 🧪 Test Case Transition & Regression Diff")
    diff_tab1, diff_tab2, diff_tab3, diff_tab4 = st.tabs([
        f"🚨 Newly Failing ({len(report.newly_failing_tests)})",
        f"🌟 Newly Passing ({len(report.newly_passing_tests)})",
        f"❌ Persistent Failures ({len(report.persistent_failing_tests)})",
        f"✅ Persistent Passes ({len(report.persistent_passing_tests)})",
    ])

    with diff_tab1:
        if report.newly_failing_tests:
            st.error(f"⚠️ **{len(report.newly_failing_tests)} test case{'s' if len(report.newly_failing_tests) != 1 else ''} passed in {report.baseline_label} but regressed and failed in {report.target_label}**")
            nf_df = pd.DataFrame([t.to_dict() for t in report.newly_failing_tests])
            st.dataframe(nf_df[[c for c in ["task_id", "query", "baseline_score", "target_score", "target_error", "details"] if c in nf_df.columns]], width="stretch")
        else:
            st.success("🎉 Zero newly failing tests! No test regressions detected.")

    with diff_tab2:
        if report.newly_passing_tests:
            st.success(f"🌟 **{len(report.newly_passing_tests)} test case{'s' if len(report.newly_passing_tests) != 1 else ''} previously failed and are now passing!**")
            np_df = pd.DataFrame([t.to_dict() for t in report.newly_passing_tests])
            st.dataframe(np_df[[c for c in ["task_id", "query", "baseline_score", "target_score", "baseline_error", "details"] if c in np_df.columns]], width="stretch")
        else:
            st.info("No newly passing test cases in this comparison.")

    with diff_tab3:
        if report.persistent_failing_tests:
            st.warning(f"**{len(report.persistent_failing_tests)} test cases failing in both versions:**")
            pf_df = pd.DataFrame([t.to_dict() for t in report.persistent_failing_tests])
            st.dataframe(pf_df[[c for c in ["task_id", "query", "baseline_score", "target_score", "target_error"] if c in pf_df.columns]], width="stretch")
        else:
            st.success("No persistent failing test cases.")

    with diff_tab4:
        st.caption(f"**{len(report.persistent_passing_tests)} test cases consistently passed across both versions.**")
        if report.persistent_passing_tests:
            pp_df = pd.DataFrame([t.to_dict() for t in report.persistent_passing_tests])
            st.dataframe(pp_df[[c for c in ["task_id", "query", "baseline_score", "target_score"] if c in pp_df.columns]], width="stretch")
