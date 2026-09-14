"""
Section 1: Overview — AgentPulse hackathon demo dashboard.

Hero banner: Agent · Version · Evaluation Status
8 KPI cards: Overall Score, Task Success, Tool Accuracy, Correctness, Groundedness,
             Avg Latency, Total Runs, Failed Tests
Charts: Pass rate by metric (bar) + Latency trend (line)
Activity feed: 12 most recent runs with one-click → Trace / → RCA
"""

import textwrap
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from src.core.entities import EvaluationResult
from dashboard.views.common import (
    health_indicator,
    render_kpi_card,
    render_section_header,
    apply_plotly_theme,
    navigate_to,
)


# ---------------------------------------------------------------------------
# Pre-compute all per-run scores in a single groupby pass
# ---------------------------------------------------------------------------
def _compute_run_scores(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    scoring_config: ScoringConfig,
) -> Dict[Any, Dict]:
    run_scores: Dict[Any, Dict] = {}
    if filtered_runs.empty:
        return run_scores

    if not filtered_evals.empty:
        records = filtered_evals.to_dict(orient="records")
        grouped: Dict[Any, List[EvaluationResult]] = {}
        for row in records:
            rid = row["run_id"]
            if rid not in grouped:
                grouped[rid] = []
            grouped[rid].append(EvaluationResult(
                metric_name=row["metric_name"],
                score=float(row["score"]),
                passed=bool(row["passed"]),
                threshold=float(row.get("threshold", 1.0) or 1.0),
                explanation=row.get("details", "") or "",
                evaluator_type=row.get("evaluator_type", "deterministic"),
            ))
    else:
        grouped = {}

    for rid in filtered_runs["run_id"]:
        ev_objs = grouped.get(rid, [])
        run_scores[rid] = calculate_case_scores(ev_objs, scoring_config)
    return run_scores


# ---------------------------------------------------------------------------
# Hero banner — built as a single-line concatenated string to avoid the
# "indented code block" bug in Streamlit's markdown parser.
# ---------------------------------------------------------------------------
def _render_hero_banner(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    run_scores: Dict,
    total_runs: int,
    failed_count: int,
    task_success_rate: float,
    avg_quality_score: float,
):
    if not filtered_runs.empty:
        latest = filtered_runs.sort_values("created_at", ascending=False).iloc[0]
        agent_name    = str(latest.get("agent_name", "—"))
        agent_version = str(latest.get("agent_version", "—"))
        model_name    = str(latest.get("model", "—"))
    else:
        agent_name = agent_version = model_name = "—"

    # Eval status
    if failed_count == 0 and total_runs > 0:
        sc, sbg, sbrd, sdot, stxt, sglow = (
            "#10b981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.35)",
            "#10b981", "ALL PASSING", "0 0 20px rgba(16,185,129,0.25)")
    elif failed_count > 0 and (failed_count / max(total_runs, 1)) > 0.3:
        sc, sbg, sbrd, sdot, stxt, sglow = (
            "#f43f5e", "rgba(244,63,94,0.12)", "rgba(244,63,94,0.35)",
            "#f43f5e", "FAILURES DETECTED", "0 0 20px rgba(244,63,94,0.20)")
    elif failed_count > 0:
        sc, sbg, sbrd, sdot, stxt, sglow = (
            "#f59e0b", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.35)",
            "#f59e0b", "PARTIALLY PASSING", "0 0 20px rgba(245,158,11,0.20)")
    else:
        sc, sbg, sbrd, sdot, stxt, sglow = (
            "#64748b", "rgba(100,116,139,0.10)", "rgba(100,116,139,0.25)",
            "#64748b", "NO DATA", "none")

    qs_color = "#10b981" if avg_quality_score >= 80 else ("#f59e0b" if avg_quality_score >= 60 else "#f43f5e")
    ts_color = "#10b981" if task_success_rate >= 80 else ("#f59e0b" if task_success_rate >= 60 else "#f43f5e")

    html = (
        f'<div style="background:linear-gradient(135deg,rgba(2,132,199,0.10) 0%,rgba(99,102,241,0.10) 100%);'
        f'border:1px solid rgba(56,189,248,0.20);border-radius:16px;padding:22px 28px;'
        f'margin-bottom:24px;box-shadow:{sglow};">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">'

        # Left: agent identity
        f'<div>'
        f'<div style="font-size:10.5px;font-weight:700;color:#38bdf8;letter-spacing:.12em;'
        f'text-transform:uppercase;margin-bottom:6px;">ACTIVE AGENT</div>'
        f'<div style="font-size:26px;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;'
        f'font-family:\'Plus Jakarta Sans\',sans-serif;">{agent_name}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px;">'
        f'<span style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);'
        f'color:#38bdf8;font-size:11px;font-weight:700;padding:3px 10px;border-radius:9999px;'
        f'font-family:\'JetBrains Mono\',monospace;">{agent_version}</span>'
        f'<span style="color:#64748b;font-size:12px;font-family:\'JetBrains Mono\',monospace;">'
        f'{model_name}</span>'
        f'</div>'
        f'</div>'

        # Centre: eval status pill
        f'<div style="text-align:center;">'
        f'<div style="font-size:10.5px;font-weight:700;color:#94a3b8;letter-spacing:.12em;'
        f'text-transform:uppercase;margin-bottom:8px;">EVALUATION STATUS</div>'
        f'<div style="background:{sbg};border:2px solid {sbrd};border-radius:12px;padding:10px 24px;'
        f'display:inline-flex;align-items:center;gap:8px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{sdot};'
        f'box-shadow:0 0 10px {sdot};flex-shrink:0;"></span>'
        f'<span style="font-size:14px;font-weight:800;color:{sc};'
        f'font-family:\'JetBrains Mono\',monospace;letter-spacing:.06em;">{stxt}</span>'
        f'</div>'
        f'<div style="margin-top:6px;font-size:11px;color:#64748b;">'
        f'{total_runs} runs &nbsp;&middot;&nbsp; {failed_count} failed</div>'
        f'</div>'

        # Right: quality score
        f'<div style="text-align:right;">'
        f'<div style="font-size:10.5px;font-weight:700;color:#94a3b8;letter-spacing:.12em;'
        f'text-transform:uppercase;margin-bottom:8px;">QUALITY SCORE</div>'
        f'<div style="font-size:40px;font-weight:900;color:{qs_color};'
        f'font-family:\'JetBrains Mono\',monospace;letter-spacing:-0.04em;line-height:1;">'
        f'{avg_quality_score:.1f}'
        f'<span style="font-size:18px;">%</span></div>'
        f'<div style="font-size:12px;color:{ts_color};margin-top:4px;">'
        f'Task success: {task_success_rate:.1f}%</div>'
        f'<div style="font-size:10.5px;color:#475569;margin-top:2px;">80% quality gate</div>'
        f'</div>'

        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_overview(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    all_runs: pd.DataFrame,
    all_evals: pd.DataFrame,
    scoring_config: ScoringConfig,
    tc_manager: Any,
):
    render_section_header(
        title="AgentPulse — Live Evaluation Console",
        subtitle="Real-time agent observability: quality, tool accuracy, groundedness, and failure drill-down.",
        breadcrumb="AGENTPULSE // OVERVIEW",
        action_badge="ENGINE ACTIVE",
    )

    if filtered_runs.empty:
        st.info("No runs found matching active filters. Run an evaluation or adjust the sidebar filters.")
        return

    # ---------------------------------------------------------------------------
    # Single-pass scoring — reused everywhere on this page
    # ---------------------------------------------------------------------------
    run_scores = _compute_run_scores(filtered_runs, filtered_evals, scoring_config)

    total_runs    = len(filtered_runs)
    eval_pass_rate = (filtered_evals["passed"].mean() * 100) if not filtered_evals.empty else 0.0
    avg_latency   = filtered_runs["latency_ms"].mean() if not filtered_runs.empty else 0.0
    total_tokens  = int(filtered_runs["total_tokens"].sum()) if not filtered_runs.empty else 0
    total_cost    = float(filtered_runs["est_cost_usd"].sum()) if not filtered_runs.empty else 0.0

    weighted_scores = [v["weighted_score"] for v in run_scores.values()]
    task_successes  = [v["passed"] for v in run_scores.values()]
    failed_count    = sum(
        1 for v in run_scores.values()
        if not v["passed"] or v.get("critical_failed", False)
    )

    avg_quality_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    task_success_rate = (sum(task_successes) / len(task_successes) * 100) if task_successes else 0.0
    error_rate        = (failed_count / total_runs * 100) if total_runs > 0 else 0.0

    def _metric_pass_rate(patterns):
        if filtered_evals.empty or "metric_name" not in filtered_evals.columns:
            return 100.0
        mask = filtered_evals["metric_name"].apply(
            lambda m: any(p in str(m).lower() for p in patterns)
        )
        sub = filtered_evals[mask]
        return float(sub["passed"].mean() * 100) if not sub.empty else 100.0

    tool_accuracy = _metric_pass_rate(["tool_selection", "tool_accuracy", "tool_argument"])
    correctness   = _metric_pass_rate(["exact_answer", "semantic_answer", "similarity", "llm_judge", "quality"])
    groundedness  = _metric_pass_rate(["groundedness", "faithfulness", "context_groundedness", "keyword_groundedness"])

    quality_label, _, _ = health_indicator(avg_quality_score, thresholds=(85.0, 70.0), higher_is_better=True)
    task_label,    _, _ = health_indicator(task_success_rate, thresholds=(90.0, 75.0), higher_is_better=True)
    error_label,   _, _ = health_indicator(error_rate,        thresholds=(25.0, 10.0), higher_is_better=False)

    # ---------------------------------------------------------------------------
    # Hero banner
    # ---------------------------------------------------------------------------
    _render_hero_banner(
        filtered_runs, filtered_evals, run_scores,
        total_runs, failed_count, task_success_rate, avg_quality_score,
    )

    # ---------------------------------------------------------------------------
    # 8 KPI cards — 2 rows × 4 columns
    # All calls to render_kpi_card use the existing single-line approach (no indented HTML)
    # ---------------------------------------------------------------------------
    st.markdown(
        '<div style="font-size:10.5px;font-weight:700;color:#64748b;text-transform:uppercase;'
        'letter-spacing:.12em;margin-bottom:10px;">KEY METRICS</div>',
        unsafe_allow_html=True,
    )

    kpi_r1c1, kpi_r1c2, kpi_r1c3, kpi_r1c4 = st.columns(4)
    with kpi_r1c1:
        render_kpi_card(
            "Overall Score", f"{avg_quality_score:.1f}%",
            subtitle="Weighted suite quality",
            health=quality_label, icon="🎯", accent_color="#38bdf8",
        )
    with kpi_r1c2:
        render_kpi_card(
            "Task Success", f"{task_success_rate:.1f}%",
            subtitle=f"{sum(task_successes)}/{total_runs} passed",
            health=task_label, icon="🏆", accent_color="#10b981",
        )
    with kpi_r1c3:
        render_kpi_card(
            "Tool Accuracy", f"{tool_accuracy:.1f}%",
            subtitle="Tool selection & args",
            health=health_indicator(tool_accuracy, (90.0, 75.0))[0],
            icon="🛠️", accent_color="#818cf8",
        )
    with kpi_r1c4:
        render_kpi_card(
            "Correctness", f"{correctness:.1f}%",
            subtitle="Exact + semantic answer quality",
            health=health_indicator(correctness, (90.0, 70.0))[0],
            icon="✅", accent_color="#a855f7",
        )

    kpi_r2c1, kpi_r2c2, kpi_r2c3, kpi_r2c4 = st.columns(4)
    with kpi_r2c1:
        render_kpi_card(
            "Groundedness", f"{groundedness:.1f}%",
            subtitle="Faithfulness & context grounding",
            health=health_indicator(groundedness, (90.0, 70.0))[0],
            icon="⚓", accent_color="#06b6d4",
        )
    with kpi_r2c2:
        render_kpi_card(
            "Avg Latency", f"{avg_latency:.0f} ms",
            subtitle="Mean execution duration",
            icon="⏱️", accent_color="#f59e0b",
        )
    with kpi_r2c3:
        render_kpi_card(
            "Total Runs", str(total_runs),
            subtitle="Evaluations executed",
            icon="🔢", accent_color="#38bdf8",
        )
    with kpi_r2c4:
        render_kpi_card(
            "Failed Tests", str(failed_count),
            subtitle=f"{error_rate:.1f}% failure rate",
            health=error_label if failed_count > 0 else "🟢 Healthy",
            icon="🚨", accent_color="#f43f5e" if failed_count > 0 else "#10b981",
        )

    st.markdown("<div style='margin-top:8px;margin-bottom:24px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Charts — single-line header strings to avoid indented-code-block bug
    # ---------------------------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
                '<div style="font-size:14px;font-weight:700;color:#f8fafc;">📊 Pass Rate by Dimension</div>'
                '<span style="font-size:9.5px;font-family:\'JetBrains Mono\',monospace;font-weight:700;'
                'color:#38bdf8;background:rgba(56,189,248,0.10);padding:2px 7px;'
                'border-radius:9999px;border:1px solid rgba(56,189,248,0.22);">BENCHMARK</span>'
                '</div>'
                '<div style="font-size:11.5px;color:#64748b;margin-bottom:10px;">'
                'Which metrics meet threshold vs. need engineering focus?</div>',
                unsafe_allow_html=True,
            )
            if not filtered_evals.empty:
                metric_summary = (
                    filtered_evals.groupby("metric_name")["passed"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                metric_summary["pass_rate_pct"] = metric_summary["mean"] * 100
                metric_summary = metric_summary.sort_values("pass_rate_pct", ascending=True)

                fig_bar = px.bar(
                    metric_summary, x="pass_rate_pct", y="metric_name",
                    orientation="h",
                    labels={"pass_rate_pct": "Pass Rate (%)", "metric_name": "Metric"},
                    range_x=[0, 100],
                    color="pass_rate_pct",
                    color_continuous_scale=[(0.0, "#f43f5e"), (0.7, "#f59e0b"), (1.0, "#10b981")],
                    range_color=[0, 100],
                )
                fig_bar.update_layout(coloraxis_showscale=False)
                fig_bar.update_traces(marker_line_width=0, opacity=0.9)
                fig_bar.add_vline(
                    x=80, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                    annotation_text="80%", annotation_position="top",
                    annotation_font_color="#f59e0b",
                )
                apply_plotly_theme(fig_bar, height=280)
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("No evaluation data yet.")

    with chart_col2:
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
                '<div style="font-size:14px;font-weight:700;color:#f8fafc;">⏱️ Latency Trend Over Time</div>'
                '<span style="font-size:9.5px;font-family:\'JetBrains Mono\',monospace;font-weight:700;'
                'color:#a855f7;background:rgba(168,85,247,0.10);padding:2px 7px;'
                'border-radius:9999px;border:1px solid rgba(168,85,247,0.22);">TIME SERIES</span>'
                '</div>'
                '<div style="font-size:11.5px;color:#64748b;margin-bottom:10px;">'
                'Is response latency stable or degrading across evaluations?</div>',
                unsafe_allow_html=True,
            )
            runs_sorted = filtered_runs.sort_values("created_at")
            if len(runs_sorted) > 500:
                runs_sorted = runs_sorted.iloc[::max(1, len(runs_sorted) // 500)]
            if len(runs_sorted) >= 2:
                # Group by timestamp and agent to eliminate duplicate timestamp points and zigzag artifacts
                chart_runs = runs_sorted.groupby(["created_at", "agent_name"], as_index=False)["latency_ms"].mean()
                fig_trend = px.line(
                    chart_runs, x="created_at", y="latency_ms", color="agent_name",
                    markers=True,
                    labels={"created_at": "Time", "latency_ms": "Latency (ms)", "agent_name": "Agent"},
                    color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f59e0b"],
                )
                fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=5))
                apply_plotly_theme(fig_trend, height=280)
                st.plotly_chart(fig_trend, width="stretch")
            else:
                st.info("Log at least 2 runs to see the latency trend.")

    st.markdown(
        "<div style='margin-top:16px;margin-bottom:22px;border-bottom:1px solid rgba(255,255,255,0.07);'></div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Activity Feed — latest 12 runs, reuses cached run_scores
    # ---------------------------------------------------------------------------
    feed_c1, feed_c2 = st.columns([3, 1])
    with feed_c1:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#f8fafc;margin-bottom:4px;">⚡ Live Run Feed</div>'
            '<div style="font-size:11.5px;color:#64748b;margin-bottom:12px;">'
            'Most recent 12 evaluations — click any row to inspect</div>',
            unsafe_allow_html=True,
        )
    with feed_c2:
        if failed_count > 0:
            if st.button(
                f"🚨 View {failed_count} Failed →",
                key="ov_goto_failures", type="primary",
                help="Go to Failures section",
            ):
                navigate_to("🚨 Failures")

    recent_runs = filtered_runs.sort_values("created_at", ascending=False).head(12)

    for _, row in recent_runs.iterrows():
        rid = row["run_id"]
        c_summary = run_scores.get(rid, {"weighted_score": 0.0, "passed": True})
        is_pass = c_summary["passed"]

        status_pill = (
            '<span style="background:rgba(16,185,129,0.13);color:#34d399;font-weight:700;'
            'font-size:10.5px;padding:2px 8px;border-radius:9999px;'
            'border:1px solid rgba(16,185,129,0.28);">✓ PASS</span>'
            if is_pass else
            '<span style="background:rgba(244,63,94,0.13);color:#fb7185;font-weight:700;'
            'font-size:10.5px;padding:2px 8px;border-radius:9999px;'
            'border:1px solid rgba(244,63,94,0.28);">✗ FAIL</span>'
        )

        with st.container(border=True):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.0, 2.2, 2.8, 1.1, 1.1, 1.1, 1.8])
            with c1:
                st.markdown(
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
                    f'font-weight:700;color:#f8fafc;">#{rid}</span><br>{status_pill}',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'🤖 **{row.get("agent_name", "Agent")}**<br>'
                    f'<span style="font-size:11px;color:#94a3b8;'
                    f'font-family:\'JetBrains Mono\',monospace;">'
                    f'{row.get("agent_version", "v1.0")} · {row.get("model", "—")}</span>',
                    unsafe_allow_html=True,
                )
            with c3:
                query_txt = str(row.get("query", ""))
                snippet = (query_txt[:52] + "…") if len(query_txt) > 52 else query_txt
                st.markdown(
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:600;'
                    f'color:#38bdf8;font-size:11.5px;">{row.get("task_id", "task")}</span><br>'
                    f'<span style="font-size:11px;color:#94a3b8;">{snippet}</span>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'⏱️ **{row.get("latency_ms", 0):.0f}ms**<br>'
                    f'<span style="font-size:10.5px;color:#64748b;">latency</span>',
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f'🎯 **{c_summary["weighted_score"]:.0f}%**<br>'
                    f'<span style="font-size:10.5px;color:#64748b;">score</span>',
                    unsafe_allow_html=True,
                )
            with c6:
                st.markdown(
                    f'🪙 **${row.get("est_cost_usd", 0):.4f}**<br>'
                    f'<span style="font-size:10.5px;color:#64748b;">'
                    f'{row.get("total_tokens", 0)} tok</span>',
                    unsafe_allow_html=True,
                )
            with c7:
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("🔍", key=f"ov_tr_{rid}", help="View execution trace"):
                        navigate_to("🔍 Traces", selected_run_id=rid)
                with btn_c2:
                    if not is_pass:
                        if st.button("🚨", key=f"ov_rca_{rid}", help="Go to Failure RCA"):
                            navigate_to("🚨 Failures", selected_run_id=rid)
                    else:
                        st.caption("✅")
