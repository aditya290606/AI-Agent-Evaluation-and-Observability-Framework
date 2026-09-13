"""
Section 6: Metrics - Per-metric pass rates, score distributions, cost vs latency scatter, and drill-downs.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.views.common import navigate_to, render_kpi_card, render_section_header, apply_plotly_theme


def render_metrics(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
):
    render_section_header(
        title="Metric Benchmarks & Distributions",
        subtitle="Detailed breakdown of assertion pass rates, score distributions, and latency-vs-cost correlations.",
        breadcrumb="OBSERVABILITY // METRICS",
        action_badge="ACCURACY ENGINE",
    )

    if filtered_evals.empty:
        st.info("No evaluation assertion metrics found matching current filters.")
        return

    # ---------------------------------------------------------------------------
    # Top KPI summary for metrics
    # ---------------------------------------------------------------------------
    unique_metrics = filtered_evals["metric_name"].unique().tolist()
    total_evals = len(filtered_evals)
    overall_pass_rate = (filtered_evals["passed"].mean() * 100)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Tracked Metrics", str(len(unique_metrics)), subtitle="Distinct Assertions", icon="📊", accent_color="#38bdf8")
    with m2:
        render_kpi_card("Total Checks", str(total_evals), subtitle="Across Filtered Runs", icon="🧪", accent_color="#818cf8")
    with m3:
        render_kpi_card("Overall Pass Rate", f"{overall_pass_rate:.1f}%", health="🟢 Healthy" if overall_pass_rate >= 90 else ("🟡 Degraded" if overall_pass_rate >= 75 else "🔴 Critical"), icon="✅", accent_color="#10b981")
    with m4:
        failing_metrics_count = sum(1 for m in unique_metrics if filtered_evals[filtered_evals["metric_name"] == m]["passed"].mean() < 0.75)
        render_kpi_card("Underperforming Metrics", str(failing_metrics_count), subtitle="Pass Rate < 75%", health="🟢 Healthy" if failing_metrics_count == 0 else "🔴 Critical", icon="⚠️", accent_color="#f43f5e" if failing_metrics_count > 0 else "#10b981")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.markdown(
                """ <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"> <div> <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">📊 Pass Rate by Evaluation Metric</div> <div style="font-size: 11.5px; color: #94a3b8;">Which evaluation dimensions fail most frequently?</div> </div> <span style="font-size: 10px; font-family: 'JetBrains Mono', monospace; background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 600;">PASS RATES</span> </div> """,
                unsafe_allow_html=True,
            )
            metric_summary = filtered_evals.groupby("metric_name")["passed"].agg(["mean", "count"]).reset_index()
            metric_summary["pass_rate_pct"] = metric_summary["mean"] * 100
            metric_summary = metric_summary.sort_values("pass_rate_pct", ascending=True)

            fig_bar = px.bar(
                metric_summary,
                x="pass_rate_pct",
                y="metric_name",
                orientation="h",
                labels={"pass_rate_pct": "Pass Rate (%)", "metric_name": "Metric"},
                range_x=[0, 100],
                color="pass_rate_pct",
                color_continuous_scale=[(0.0, "#f43f5e"), (0.7, "#f59e0b"), (1.0, "#10b981")],
                range_color=[0, 100],
            )
            fig_bar.update_layout(coloraxis_showscale=False)
            fig_bar.update_traces(marker_line_width=0, opacity=0.9)
            apply_plotly_theme(fig_bar, height=300)
            st.plotly_chart(fig_bar, width="stretch")

    with col_right:
        with st.container(border=True):
            st.markdown(
                """ <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"> <div> <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">🪙 Cost vs. Latency Distribution</div> <div style="font-size: 11.5px; color: #94a3b8;">Are slower runs incurring higher token costs?</div> </div> <span style="font-size: 10px; font-family: 'JetBrains Mono', monospace; background: rgba(99, 102, 241, 0.12); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 600;">CORRELATION</span> </div> """,
                unsafe_allow_html=True,
            )
            fig_scatter = px.scatter(
                filtered_runs,
                x="latency_ms",
                y="est_cost_usd",
                color="agent_name",
                hover_data=["run_id", "task_id", "agent_name", "tools_called"],
                labels={"latency_ms": "Latency (ms)", "est_cost_usd": "Est Cost (USD)", "agent_name": "Agent"},
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f59e0b"],
            )
            apply_plotly_theme(fig_scatter, height=300)
            st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("<div style='margin-top: 14px; margin-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Metric Deep-Dive & Score Distribution
    # ---------------------------------------------------------------------------
    st.markdown("### 🔍 Metric Drill-Down & Score Distributions")
    st.caption("Select an evaluation metric to inspect its score distribution and jump to underlying runs.")

    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        chosen_metric = st.selectbox("Select Metric to Inspect:", sorted(unique_metrics), key="metric_drilldown_selector")
    with d_col2:
        st.write("")
        st.write("")
        if st.button("🚀 View Underlying Runs ➔", key="btn_jump_runs_for_m"):
            navigate_to("🚀 Evaluation Runs", global_metric_filter=[chosen_metric])

    metric_evals = filtered_evals[filtered_evals["metric_name"] == chosen_metric]

    if not metric_evals.empty:
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            render_kpi_card("Metric Pass Rate", f"{metric_evals['passed'].mean() * 100:.1f}%", icon="🎯", accent_color="#38bdf8")
        with stat_c2:
            render_kpi_card("Average Score", f"{metric_evals['score'].mean():.2f}", icon="📈", accent_color="#818cf8")
        with stat_c3:
            render_kpi_card("Min / Max Score", f"{metric_evals['score'].min():.2f} / {metric_evals['score'].max():.2f}", icon="📏", accent_color="#10b981")
        with stat_c4:
            render_kpi_card("Total Checks", str(len(metric_evals)), icon="🔢", accent_color="#f59e0b")

        with st.container(border=True):
            st.markdown(f"**Score Distribution for '{chosen_metric}'**")
            fig_hist = px.histogram(
                metric_evals,
                x="score",
                nbins=15,
                color="passed",
                labels={"score": "Score (0.0 - 1.0)", "passed": "Passed?"},
                color_discrete_map={True: "#10b981", False: "#f43f5e"},
            )
            apply_plotly_theme(fig_hist, height=260)
            st.plotly_chart(fig_hist, width="stretch")

        st.markdown(f"#### Recent Checks for `{chosen_metric}`")
        recent_checks = metric_evals.head(8)
        for _, chk in recent_checks.iterrows():
            passed = bool(chk.get("passed", False))
            c_badge = "#10b981" if passed else "#f43f5e"
            c_text = "PASSED" if passed else "FAILED"
            with st.container(border=True):
                r_c1, r_c2, r_c3 = st.columns([3, 2, 5])
                with r_c1:
                    st.markdown(f"**Run #{chk['run_id']}** (`{chk['task_id']}`)")
                with r_c2:
                    st.markdown(
                        f""" <span style="background: {c_badge}18; color: {c_badge}; border: 1px solid {c_badge}40; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700;"> {c_text} ({chk['score']:.2f}) </span> """,
                        unsafe_allow_html=True,
                    )
                with r_c3:
                    detail_str = str(chk.get("details", "") or chk.get("explanation", ""))
                    st.caption(detail_str[:120] + ("..." if len(detail_str) > 120 else "") if detail_str else "Evaluation passed assertion threshold.")
