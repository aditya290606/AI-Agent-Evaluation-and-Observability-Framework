"""
Section 5: Traces — Full Observability Feature.

Answers five core questions for every SDK execution:
  1. What did the agent do?
  2. Which tool did it call?
  3. Where did it fail?
  4. Which step was slow?
  5. What was the final result?

Renders:
  • Run header (agent, task, model, tokens, cost)
  • Observability KPI cards
  • Execution Timeline (Gantt chart)
  • Hierarchical Trace Tree (expandable)
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.tracing.tracer import build_trace_tree, Span
from src.core.entities import EvaluationResult, Trace
from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from dashboard.views.common import (
    steps_to_spans,
    navigate_to,
    render_section_header,
    apply_plotly_theme,
    render_failure_analysis,
)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "agent": "#818cf8",
    "llm": "#38bdf8",
    "tool": "#34d399",
    "mcp_tool": "#a855f7",
    "mcp_server": "#c084fc",
    "retrieval": "#f59e0b",
    "embedding": "#06b6d4",
    "memory": "#e879f9",
    "planning": "#64748b",
    "final_answer": "#10b981",
    "error": "#f43f5e",
}

TYPE_ICONS = {
    "agent": "🤖",
    "llm": "🧠",
    "tool": "🛠️",
    "mcp_tool": "🔌",
    "mcp_server": "🌐",
    "retrieval": "📚",
    "embedding": "📐",
    "memory": "💾",
    "planning": "📋",
    "final_answer": "💬",
    "error": "⚠️",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 120) -> str:
    if not text:
        return ""
    return (text[:max_len] + "…") if len(text) > max_len else text


def _format_tokens(inp: int, out: int) -> str:
    total = inp + out
    if total == 0:
        return "—"
    return f"{inp:,} in / {out:,} out ({total:,} total)"


def _format_cost(cost: float) -> str:
    if cost <= 0:
        return "—"
    return f"${cost:.5f}"


def _safe_json_preview(data: str, max_lines: int = 20) -> str:
    """Try to pretty-print JSON; fall back to raw text."""
    if not data:
        return ""
    try:
        parsed = json.loads(data)
        pretty = json.dumps(parsed, indent=2, default=str)
        lines = pretty.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + "\n  ... (truncated)"
        return pretty
    except Exception:
        return data


def _detect_language(text: str) -> str:
    if not text:
        return "text"
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return "json"
    if stripped.startswith(("def ", "import ", "class ", "from ")):
        return "python"
    return "text"


# ---------------------------------------------------------------------------
# Observability analysis helpers (non-fabricated)
# ---------------------------------------------------------------------------

def _analyze_spans(spans: List[Span]) -> Dict[str, Any]:
    """Derive observability answers from real span data.  Never fabricate."""
    analysis: Dict[str, Any] = {
        "total_spans": len(spans),
        "span_types": {},
        "tools_called": [],
        "llm_calls": [],
        "errors": [],
        "slowest_span": None,
        "final_answer_span": None,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_cost": 0.0,
        "total_duration_ms": 0.0,
    }

    for sp in spans:
        stype = sp.span_type
        analysis["span_types"][stype] = analysis["span_types"].get(stype, 0) + 1

        if stype in ("tool", "mcp_tool") and sp.tool_name:
            analysis["tools_called"].append({
                "name": sp.tool_name,
                "status": sp.status,
                "duration_ms": sp.latency_ms,
                "mcp_server": sp.mcp_server,
            })

        if stype == "llm":
            analysis["llm_calls"].append({
                "model": sp.model or "unknown",
                "tokens_in": sp.input_tokens,
                "tokens_out": sp.output_tokens,
                "duration_ms": sp.latency_ms,
            })

        if sp.status != "success" or sp.error:
            analysis["errors"].append({
                "span_id": sp.span_id,
                "operation": sp.operation_name,
                "type": stype,
                "error": sp.error or "Operation failed",
                "duration_ms": sp.latency_ms,
            })

        if stype == "final_answer":
            analysis["final_answer_span"] = sp

        if analysis["slowest_span"] is None or sp.latency_ms > analysis["slowest_span"].latency_ms:
            analysis["slowest_span"] = sp

        analysis["total_tokens_in"] += sp.input_tokens
        analysis["total_tokens_out"] += sp.output_tokens
        analysis["total_cost"] += sp.cost_usd
        analysis["total_duration_ms"] += sp.latency_ms

    return analysis


def _glass_card(title: str, value: str, detail: str, accent: str, icon: str) -> str:
    """Render a single glass KPI card HTML."""
    return (
        f'<div style="background: linear-gradient(135deg, rgba(17,24,39,0.85), rgba(15,23,42,0.95)); '
        f'border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 18px; '
        f'box-shadow: 0 4px 20px -2px rgba(0,0,0,0.4); position: relative; overflow: hidden; min-height: 110px;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{accent},transparent);"></div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">'
        f'<span style="font-size:16px;">{icon}</span>'
        f'<span style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">{title}</span>'
        f'</div>'
        f'<div style="font-size:18px;font-weight:800;color:#f8fafc;font-family:\'JetBrains Mono\',monospace;margin-bottom:4px;">{value}</div>'
        f'<div style="font-size:11px;color:#64748b;font-weight:500;">{detail}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_traces(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    load_steps_fn: Any,
    scoring_config: ScoringConfig,
    tc_manager: Optional[Any] = None,
):
    render_section_header(
        title="Execution Traces & Span Hierarchy",
        subtitle="Deep-dive observability into agent reasoning, tool execution latencies, token consumption, and nested span trees.",
        breadcrumb="OBSERVABILITY // TRACES",
        action_badge="SPAN TRACER",
    )

    if filtered_runs.empty:
        st.info("No runs match the active filters. Adjust your global filters or run an evaluation.")
        return

    # ------------------------------------------------------------------
    # Run selector
    # ------------------------------------------------------------------
    all_run_ids = filtered_runs["run_id"].tolist()
    default_run_id = st.session_state.get("selected_run_id", all_run_ids[0])
    if default_run_id not in all_run_ids:
        default_run_id = all_run_ids[0]

    run_label_map = {
        row["run_id"]: f"Run #{row['run_id']} — Task: {row.get('task_id', '')} ({str(row.get('query', ''))[:45]}...)"
        for row in filtered_runs.to_dict(orient="records")
    }
    sel_c1, sel_c2 = st.columns([3, 1])
    with sel_c1:
        selected_run_id = st.selectbox(
            "Select Run ID to Inspect",
            all_run_ids,
            index=all_run_ids.index(default_run_id),
            format_func=lambda x: run_label_map.get(x, f"Run #{x}"),
            key="trace_run_selector"
        )
    with sel_c2:
        st.write("")
        st.write("")
        if st.button("🚨 View Root Cause", key="btn_trace_to_rca", help="Jump to AI Root Cause Analysis for this run"):
            navigate_to("🚨 Failures", selected_run_id=selected_run_id)

    run_row = filtered_runs[filtered_runs["run_id"] == selected_run_id].iloc[0]
    run_evals = filtered_evals[filtered_evals["run_id"] == selected_run_id] if not filtered_evals.empty else pd.DataFrame()

    # ------------------------------------------------------------------
    # Run header card
    # ------------------------------------------------------------------
    st.markdown(
        f""" <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.4);"> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 10px;"> <div style="font-size: 16px; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;"> <span>Run #{selected_run_id}</span> <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 11px; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">{run_row.get('model', 'claude-3-5-haiku')}</span> <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 11px; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">{'Mock Mode' if run_row.get('is_mock') else 'Live Mode'}</span> </div> <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8;"> ⏱️ <strong style="color: #f8fafc;">{run_row.get('latency_ms', 0):.1f} ms</strong> &nbsp;|&nbsp; 🪙 <strong style="color: #34d399;">${run_row.get('est_cost_usd', 0):.5f}</strong> &nbsp;|&nbsp; 🔢 <strong style="color: #a855f7;">{run_row.get('total_tokens', 0):,} tok</strong> </div> </div> <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 13px;"> <div> <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;">Agent & Task ID</div> <div style="color: #f1f5f9; font-weight: 600; margin-top: 2px;">🤖 {run_row.get('agent_name', 'DemoAgent')} <span style="color: #64748b;">({run_row.get('agent_version', 'v1.0')})</span> · 🧪 <span style="color: #38bdf8; font-family: monospace;">{run_row['task_id']}</span></div> <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 10px;">User Query Input</div> <div style="color: #cbd5e1; margin-top: 2px; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px; font-family: monospace; font-size: 12px;">{run_row['query']}</div> </div> <div> <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;">Experiment / Environment</div> <div style="color: #f1f5f9; margin-top: 2px;"><span style="color: #a78bfa; font-family: monospace;">{run_row.get('experiment_id', 'legacy')}</span> · <span style="color: #64748b;">{run_row.get('environment', 'production')}</span></div> <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 10px;">Final Agent Answer</div> <div style="color: #e2e8f0; margin-top: 2px; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px; font-size: 12px; max-height: 80px; overflow-y: auto;">{run_row.get('final_answer', '—')}</div> </div> </div> </div> """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Load spans
    # ------------------------------------------------------------------
    steps = load_steps_fn(int(selected_run_id))
    span_objects = steps_to_spans(steps)

    if not span_objects:
        st.info("No spans recorded for this run.")
        return

    analysis = _analyze_spans(span_objects)

    # ------------------------------------------------------------------
    # Observability KPI cards — answers the 5 core questions
    # ------------------------------------------------------------------
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;margin-top:4px;">QUICK OBSERVABILITY ANSWERS</div>',
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(5)

    # Q1: What did the agent do?
    type_summary_parts = []
    for stype, count in sorted(analysis["span_types"].items(), key=lambda x: -x[1]):
        type_summary_parts.append(f"{count}× {stype}")
    type_summary = ", ".join(type_summary_parts[:4])
    with kpi_cols[0]:
        st.markdown(
            _glass_card(
                "What did it do?",
                f"{analysis['total_spans']} spans",
                type_summary or "No operations recorded",
                "#818cf8",
                "🤖",
            ),
            unsafe_allow_html=True,
        )

    # Q2: Which tool did it call?
    tools = analysis["tools_called"]
    if tools:
        tool_names = [t["name"] for t in tools]
        unique_tools = list(dict.fromkeys(tool_names))  # preserve order, deduplicate
        tool_display = ", ".join(unique_tools[:3])
        if len(unique_tools) > 3:
            tool_display += f" +{len(unique_tools) - 3}"
    else:
        tool_display = "No tools called"
    with kpi_cols[1]:
        st.markdown(
            _glass_card(
                "Which tools?",
                f"{len(tools)} call{'s' if len(tools) != 1 else ''}",
                tool_display,
                "#34d399",
                "🛠️",
            ),
            unsafe_allow_html=True,
        )

    # Q3: Where did it fail?
    errors = analysis["errors"]
    if errors:
        first_err = errors[0]
        err_detail = _truncate(first_err["error"], 50)
        err_display = f"{first_err['operation']}: {err_detail}"
    else:
        err_display = "No failures detected"
    with kpi_cols[2]:
        err_accent = "#f43f5e" if errors else "#10b981"
        st.markdown(
            _glass_card(
                "Where failed?",
                f"{len(errors)} error{'s' if len(errors) != 1 else ''}" if errors else "✓ Clean",
                err_display,
                err_accent,
                "🔴" if errors else "🟢",
            ),
            unsafe_allow_html=True,
        )

    # Q4: Which step was slow?
    slowest = analysis["slowest_span"]
    if slowest:
        slow_name = slowest.tool_name or slowest.operation_name
        slow_display = f"{_truncate(slow_name, 30)} ({slowest.latency_ms:.0f}ms)"
    else:
        slow_display = "No spans"
    with kpi_cols[3]:
        st.markdown(
            _glass_card(
                "Slowest step?",
                f"{slowest.latency_ms:.0f}ms" if slowest else "—",
                slow_display,
                "#f59e0b",
                "🐢",
            ),
            unsafe_allow_html=True,
        )

    # Q5: What was the final result?
    final_answer = run_row.get("final_answer", "")
    if final_answer:
        answer_preview = _truncate(str(final_answer), 60)
    else:
        fa_span = analysis.get("final_answer_span")
        if fa_span and fa_span.output_data:
            answer_preview = _truncate(fa_span.output_data, 60)
        else:
            answer_preview = "No final answer recorded"

    with kpi_cols[4]:
        st.markdown(
            _glass_card(
                "Final result?",
                "✓ Answered" if final_answer else "⚠️ No answer",
                answer_preview,
                "#10b981" if final_answer else "#f59e0b",
                "💬",
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Execution Flow Summary
    # ------------------------------------------------------------------
    st.markdown("")
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px;">EXECUTION FLOW</div>',
        unsafe_allow_html=True,
    )

    # Build a compact flow string: Agent → LLM → Tool(search) → LLM → Final Answer
    flow_parts = []
    for sp in span_objects:
        stype = sp.span_type
        icon = TYPE_ICONS.get(stype, "🔹")
        if stype in ("tool", "mcp_tool") and sp.tool_name:
            flow_parts.append(f"{icon} {sp.tool_name}")
        elif stype == "llm":
            model_tag = f"({sp.model})" if sp.model else ""
            flow_parts.append(f"{icon} LLM {model_tag}")
        elif stype == "final_answer":
            flow_parts.append(f"{icon} Answer")
        elif stype == "retrieval":
            flow_parts.append(f"{icon} RAG")
        elif stype == "agent":
            flow_parts.append(f"{icon} Agent")
        elif stype == "error":
            flow_parts.append(f"{icon} Error")
        else:
            flow_parts.append(f"{icon} {stype.capitalize()}")

    flow_html_parts = []
    for i, part in enumerate(flow_parts):
        flow_html_parts.append(
            f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);'
            f'padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;color:#e2e8f0;'
            f'font-family:\'JetBrains Mono\',monospace;white-space:nowrap;">{part}</span>'
        )
        if i < len(flow_parts) - 1:
            flow_html_parts.append(
                '<span style="color:#475569;font-size:14px;margin:0 2px;">→</span>'
            )

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:12px 14px;'
        f'background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;'
        f'margin-bottom:20px;overflow-x:auto;">{"".join(flow_html_parts)}</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Failure Analysis (if run failed evaluation or has errors)
    # ------------------------------------------------------------------
    run_evals = filtered_evals[filtered_evals["run_id"] == selected_run_id] if not filtered_evals.empty else pd.DataFrame()
    eval_objs = []
    has_failed_eval = False
    if not run_evals.empty:
        for _, erow in run_evals.iterrows():
            eo = EvaluationResult(
                metric_name=erow["metric_name"],
                score=float(erow["score"]),
                passed=bool(erow["passed"]),
                threshold=float(erow.get("threshold", 1.0) or 1.0),
                explanation=erow.get("details", "") or erow.get("explanation", ""),
            )
            eval_objs.append(eo)
            if not eo.passed:
                has_failed_eval = True

    has_errors = bool(analysis.get("errors"))
    if has_failed_eval or has_errors:
        with st.expander("🔬 Failure Analysis & Root Cause Diagnosis", expanded=True):
            t_case = tc_manager.get_test_case(run_row.get("task_id", "")) if tc_manager else None
            trace_obj = Trace(
                task_id=run_row.get("task_id", ""),
                query=run_row.get("query", ""),
                final_answer=run_row.get("final_answer", ""),
                spans=span_objects,
                is_mock=bool(run_row.get("is_mock", True)),
            )
            trace_obj.latency_ms = float(run_row.get("latency_ms") or 0.0)
            render_failure_analysis(t_case, trace_obj, eval_objs, show_all=True)

    # ------------------------------------------------------------------
    # 1. Execution Timeline (Gantt Bottleneck Analysis)
    # ------------------------------------------------------------------
    st.markdown("### ⏱️ Execution Timeline")
    st.caption("Horizontal bars show each operation's duration relative to run start. Color = span type. Red = error.")

    min_t = min((s.start_time for s in span_objects if s.start_time and s.start_time > 0), default=0.0)
    y_labels = []
    bases = []
    durations = []
    colors = []
    hover_texts = []
    border_colors = []

    for idx, sp in enumerate(span_objects):
        offset_ms = (sp.start_time - min_t) * 1000 if (sp.start_time and min_t) else (idx * 5.0)
        dur = max(sp.latency_ms, 0.5)

        # Y-axis label
        if sp.is_mcp or sp.span_type == "mcp_tool":
            srv_str = f" [{sp.mcp_server}]" if sp.mcp_server else ""
            t_name = sp.tool_name or sp.operation_name
            y_name = f"🔌 MCP{srv_str}: {t_name[:20]}"
        elif sp.span_type == "mcp_server":
            y_name = f"🌐 MCP SERVER: {sp.operation_name[:24]}"
        elif sp.span_type == "tool" and sp.tool_name:
            y_name = f"🛠️ TOOL: {sp.tool_name[:24]}"
        elif sp.span_type == "llm":
            model_name = sp.model or "LLM"
            y_name = f"🧠 LLM: {model_name[:24]}"
        elif sp.span_type == "retrieval":
            y_name = f"📚 RAG: {sp.operation_name[:24]}"
        elif sp.span_type == "final_answer":
            y_name = f"💬 ANSWER: {sp.operation_name[:24]}"
        else:
            icon = TYPE_ICONS.get(sp.span_type, "🔹")
            y_name = f"{icon} {sp.span_type.upper()}: {sp.operation_name[:28]}"

        y_labels.append(y_name)
        bases.append(offset_ms)
        durations.append(dur)

        # Color — red border for errors
        bar_color = TYPE_COLORS.get(sp.span_type, "#a855f7" if sp.is_mcp else "#94a3b8")
        if sp.status != "success" or sp.error:
            bar_color = "#f43f5e"
        colors.append(bar_color)

        # Hover
        tool_extra = ""
        if sp.is_mcp or sp.span_type == "mcp_tool":
            inp_preview = _truncate(sp.input_data, 50)
            tool_extra = f"<br>Protocol: MCP<br>Server: {sp.mcp_server or 'Unknown'}<br>Tool: {sp.tool_name or sp.operation_name}<br>Args: {inp_preview}"
        elif sp.span_type == "tool" and sp.tool_name:
            inp_preview = _truncate(sp.input_data, 50)
            tool_extra = f"<br>Tool: {sp.tool_name}<br>Args: {inp_preview}"
        elif sp.span_type == "llm":
            tool_extra = f"<br>Model: {sp.model or 'unknown'}<br>Tokens: {sp.input_tokens} in / {sp.output_tokens} out"
        if sp.error:
            tool_extra += f"<br><span style='color:#f43f5e;'>Error: {_truncate(sp.error, 50)}</span>"

        hover_texts.append(
            f"Operation: {sp.operation_name}<br>"
            f"Type: {sp.span_type}<br>"
            f"Offset: {offset_ms:.1f}ms<br>"
            f"Duration: {dur:.1f}ms<br>"
            f"Status: {sp.status}{tool_extra}<br>"
            f"Cost: {_format_cost(sp.cost_usd)}<br>"
            f"Tokens: {sp.input_tokens} in / {sp.output_tokens} out"
        )

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(
        y=y_labels,
        x=durations,
        base=bases,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=1, color=[c if c != "#f43f5e" else "#ff6b81" for c in colors]),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig_timeline.update_layout(
        height=max(250, len(y_labels) * 40),
        xaxis_title="Time from Run Start (ms)",
        yaxis=dict(autorange="reversed"),
    )
    apply_plotly_theme(fig_timeline, height=max(250, len(y_labels) * 40))
    st.plotly_chart(fig_timeline, width="stretch")

    # ------------------------------------------------------------------
    # 2. Hierarchical Trace Tree
    # ------------------------------------------------------------------
    st.markdown("### 🌳 Hierarchical Trace Tree")
    st.caption("Expand nodes to inspect inputs, outputs, tokens, cost, and metadata. Failed spans are marked in red. MCP spans highlighted with 🔌.")

    tree_roots = build_trace_tree(span_objects)

    def _status_badge(status: str, error: Optional[str] = None) -> str:
        if error or status != "success":
            return (
                '<span style="background:rgba(244,63,94,0.15);color:#fb7185;padding:2px 8px;border-radius:9999px;'
                'font-size:10px;font-weight:700;border:1px solid rgba(244,63,94,0.3);">ERROR</span>'
            )
        return (
            '<span style="background:rgba(16,185,129,0.15);color:#34d399;padding:2px 8px;border-radius:9999px;'
            'font-size:10px;font-weight:700;border:1px solid rgba(16,185,129,0.3);">OK</span>'
        )

    def render_tree_node(node: Span, depth: int = 0, run_id: Any = None):
        icon = TYPE_ICONS.get(node.span_type, "🔌" if node.is_mcp else "🔹")
        dur_text = f"{node.duration_ms:.1f}ms"
        cost_text = _format_cost(node.cost_usd)

        if node.is_mcp or node.span_type == "mcp_tool":
            type_label = f"MCP TOOL // {node.mcp_server}" if node.mcp_server else "MCP TOOL"
        elif node.span_type == "mcp_server":
            type_label = f"MCP SERVER // {node.mcp_server or node.operation_name}"
        else:
            type_label = node.span_type.upper()

        # Status indicator
        status_dot = "🟢" if (node.status == "success" and not node.error) else "🔴"

        indent = "—" * (depth * 2) + " " if depth > 0 else ""
        header_label = f"{indent}{icon} [{type_label}] {node.operation_name} — {dur_text} {status_dot}"
        if node.cost_usd > 0:
            header_label += f" — {cost_text}"

        is_error = node.status != "success" or bool(node.error)
        with st.expander(header_label, expanded=(depth == 0 or is_error)):
            # Detail row using HTML for compact layout
            detail_parts = []
            detail_parts.append(f"<strong>Span:</strong> <code>{node.span_id}</code>")
            if node.parent_span_id:
                detail_parts.append(f"<strong>Parent:</strong> <code>{node.parent_span_id}</code>")
            detail_parts.append(f"<strong>Duration:</strong> <code>{dur_text}</code>")
            if node.model:
                detail_parts.append(f"<strong>Model:</strong> <code>{node.model}</code>")
            if node.tool_name:
                detail_parts.append(f"<strong>Tool:</strong> <code>{node.tool_name}</code>")
            if node.mcp_server:
                detail_parts.append(f"<strong>MCP Server:</strong> <code>{node.mcp_server}</code>")
            tokens_total = node.input_tokens + node.output_tokens
            if tokens_total > 0:
                detail_parts.append(f"<strong>Tokens:</strong> <code>{node.input_tokens}</code> in / <code>{node.output_tokens}</code> out")
            if node.cost_usd > 0:
                detail_parts.append(f"<strong>Cost:</strong> <code>{cost_text}</code>")

            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:#94a3b8;padding:4px 0 8px 0;">'
                + " &nbsp;|&nbsp; ".join(detail_parts)
                + '</div>',
                unsafe_allow_html=True,
            )

            # Error display
            if is_error:
                st.error(f"🚨 **Span Error:** {node.error or 'Operation failed without explicit exception.'}")
                if st.button("🚨 Analyze in Root Cause Analysis", key=f"btn_rca_{node.span_id}_{depth}"):
                    navigate_to("🚨 Failures", selected_run_id=run_id or selected_run_id)

            # Input/Output in two columns
            has_input = bool(node.input_data)
            has_output = bool(node.output_data)
            if has_input or has_output:
                io_cols = st.columns(2)
                with io_cols[0]:
                    if has_input:
                        st.markdown("**📥 Input:**")
                        preview = _safe_json_preview(node.input_data)
                        lang = _detect_language(preview)
                        st.code(preview, language=lang)
                    else:
                        st.caption("No input data")
                with io_cols[1]:
                    if has_output:
                        st.markdown("**📤 Output:**")
                        preview = _safe_json_preview(node.output_data)
                        lang = _detect_language(preview)
                        st.code(preview, language=lang)
                    else:
                        st.caption("No output data")

            # Metadata
            if node.attributes:
                with st.expander("📎 Metadata & Telemetry", expanded=False):
                    st.json(node.attributes)

        # Render children
        for child in node.children:
            render_tree_node(child, depth + 1, run_id=run_id)

    if tree_roots:
        for root_node in tree_roots:
            render_tree_node(root_node, depth=0, run_id=selected_run_id)
    else:
        # Fallback: if no parent_span_id hierarchy, render flat list
        st.info("No hierarchical parent-child relationships found. Showing flat span list.")
        for sp in span_objects:
            render_tree_node(sp, depth=0, run_id=selected_run_id)

    # ------------------------------------------------------------------
    # 3. Span Statistics Table
    # ------------------------------------------------------------------
    st.markdown("### 📊 Span Statistics")
    if span_objects:
        stats_data = []
        for sp in span_objects:
            stats_data.append({
                "Type": f"{TYPE_ICONS.get(sp.span_type, '🔹')} {sp.span_type}",
                "Operation": _truncate(sp.operation_name, 40),
                "Duration (ms)": round(sp.latency_ms, 1),
                "Status": "✅" if sp.status == "success" and not sp.error else "❌",
                "Tokens In": sp.input_tokens,
                "Tokens Out": sp.output_tokens,
                "Cost ($)": round(sp.cost_usd, 5) if sp.cost_usd > 0 else 0,
                "Model/Tool": sp.model or sp.tool_name or "—",
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(
            stats_df,
            width="stretch",
            hide_index=True,
            height=min(400, 40 + len(stats_data) * 35),
        )
