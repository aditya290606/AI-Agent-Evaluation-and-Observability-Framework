"""
Common UI components, helper utilities, and high-end design tokens for the Agent Eval Console.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from src.core.entities import EvaluationResult, Trace
from src.tracing.tracer import Span
from src.evaluation.scoring import ScoringConfig, calculate_case_scores


def health_indicator(
    value: float,
    thresholds: Tuple[float, float] = (90.0, 70.0),
    higher_is_better: bool = True
) -> Tuple[str, str, str]:
    """
    Returns (label, hex_color, status_slug) for health indicators:
    🟢 Healthy, 🟡 Degraded, 🔴 Critical
    """
    if higher_is_better:
        if value >= thresholds[0]:
            return "🟢 Healthy", "#10b981", "healthy"
        elif value >= thresholds[1]:
            return "🟡 Degraded", "#f59e0b", "degraded"
        else:
            return "🔴 Critical", "#f43f5e", "critical"
    else:
        if value <= thresholds[1]:
            return "🟢 Healthy", "#10b981", "healthy"
        elif value <= thresholds[0]:
            return "🟡 Degraded", "#f59e0b", "degraded"
        else:
            return "🔴 Critical", "#f43f5e", "critical"


def render_section_header(title: str, subtitle: str = "", breadcrumb: str = "CONSOLE", action_badge: Optional[str] = None):
    """
    Renders an executive hero header with glowing accent and breadcrumbs.
    """
    badge_html = ""
    if action_badge:
        badge_html = (
            f'<div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); '
            f'color: #38bdf8; font-family: \'JetBrains Mono\', monospace; font-size: 11px; padding: 3px 10px; '
            f'border-radius: 9999px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; '
            f'vertical-align: middle;">'
            f'<span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#38bdf8; '
            f'box-shadow:0 0 8px #38bdf8;"></span>{action_badge}</div>'
        )

    html = (
        f'<div style="margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.07);">'
        f'<div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;">{breadcrumb}</div>'
        f'<div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 4px;">'
        f'<h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; font-family: \'Plus Jakarta Sans\', sans-serif;">{title}</h1>'
        f'{badge_html}'
        f'</div>'
        f'<div style="font-size: 13px; color: #94a3b8;">{subtitle}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    health: Optional[str] = None,
    icon: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    accent_color: Optional[str] = None,
):
    """
    Renders an executive glassmorphic card with zero markdown indentation issues.
    """
    # Color logic
    color = accent_color or "#38bdf8"
    pill_html = ""
    if health:
        if "Healthy" in health or "🟢" in health:
            pill_html = '<span style="background: rgba(16, 185, 129, 0.12); color: #34d399; padding: 3px 8px; border-radius: 9999px; font-size: 10.5px; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.25); display: inline-flex; align-items: center; gap: 4px;"><span style="width: 5px; height: 5px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>HEALTHY</span>'
            color = "#10b981"
        elif "Degraded" in health or "🟡" in health:
            pill_html = '<span style="background: rgba(245, 158, 11, 0.12); color: #fbbf24; padding: 3px 8px; border-radius: 9999px; font-size: 10.5px; font-weight: 700; border: 1px solid rgba(245, 158, 11, 0.25); display: inline-flex; align-items: center; gap: 4px;"><span style="width: 5px; height: 5px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 6px #f59e0b;"></span>DEGRADED</span>'
            color = "#f59e0b"
        elif "Critical" in health or "🔴" in health:
            pill_html = '<span style="background: rgba(244, 63, 94, 0.12); color: #fb7185; padding: 3px 8px; border-radius: 9999px; font-size: 10.5px; font-weight: 700; border: 1px solid rgba(244, 63, 94, 0.25); display: inline-flex; align-items: center; gap: 4px;"><span style="width: 5px; height: 5px; border-radius: 50%; background: #f43f5e; box-shadow: 0 0 6px #f43f5e;"></span>CRITICAL</span>'
            color = "#f43f5e"
        else:
            pill_html = f'<span style="background: rgba(255,255,255,0.06); color: #cbd5e1; padding: 3px 8px; border-radius: 9999px; font-size: 10.5px; font-weight: 600;">{health}</span>'

    delta_html = ""
    if delta:
        delta_color_css = "#34d399" if delta_color == "normal" else "#fb7185"
        delta_html = f'<div style="font-size: 11.5px; font-weight: 600; color: {delta_color_css}; margin-top: 5px; display: flex; align-items: center; gap: 4px;">{delta}</div>'

    subtitle_html = f'<div style="font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 500;">{subtitle}</div>' if subtitle else ""
    icon_html = f'<span style="font-size: 15px; margin-right: 6px;">{icon}</span>' if icon else ""

    # Zero-indented single-string HTML to prevent markdown code block triggers
    card_html = (
        f'<div class="kpi-glass-card" style="position: relative; overflow: hidden; background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4); backdrop-filter: blur(12px);">'
        f'<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, {color}, transparent);"></div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; display: flex; align-items: center;">{icon_html}{title}</div>'
        f'{pill_html}'
        f'</div>'
        f'<div style="font-size: 26px; font-weight: 800; color: #f8fafc; font-family: \'JetBrains Mono\', monospace; letter-spacing: -0.02em;">{value}</div>'
        f'{delta_html}'
        f'{subtitle_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def apply_plotly_theme(fig, height: int = 320):
    """
    Applies high-end dark glassmorphism styling to any Plotly figure.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=16, r=16, t=24, b=24),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#64748b", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#64748b", size=10),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#334155",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#f8fafc"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            font=dict(color="#cbd5e1", size=10.5),
        ),
    )
    return fig


def navigate_to(section: str, **kwargs):
    """
    Programmatic navigation helper for seamless drill-downs.
    """
    st.session_state["nav_section"] = section
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


def steps_to_spans(steps: List[Any]) -> List[Span]:
    """
    Convert database Step objects to Span entities for trace analysis.
    """
    span_objects: List[Span] = []
    for s in steps:
        meta = {}
        if getattr(s, "metadata_json", None):
            try:
                meta = json.loads(s.metadata_json)
            except Exception:
                meta = {}

        span_obj = Span(
            step_type=s.step_type,
            span_id=s.span_id or f"spn_{s.id}",
            parent_span_id=s.parent_span_id,
            run_id=s.run_id,
            operation_name=getattr(s, "operation_name", None) or (f"Tool: {s.tool_name}" if s.tool_name else f"{s.step_type.capitalize()} Operation"),
            tool_name=s.tool_name,
            mcp_server=getattr(s, "mcp_server", None) or meta.get("mcp_server"),
            input_data=s.input_data or "",
            output_data=s.output_data or "",
            latency_ms=s.latency_ms or 0.0,
            start_time=getattr(s, "start_time", None) or 0.0,
            end_time=getattr(s, "end_time", None) or 0.0,
            status=getattr(s, "status", None) or "success",
            error=getattr(s, "error", None),
            model=getattr(s, "model", None),
            cost_usd=getattr(s, "cost_usd", 0.0) or 0.0,
            attributes=meta,
            input_tokens=s.input_tokens or 0,
            output_tokens=s.output_tokens or 0,
            step_index=s.step_index,
        )
        span_objects.append(span_obj)
    return span_objects


def compute_run_composite(run_row: pd.Series, run_evals: pd.DataFrame, scoring_config: ScoringConfig) -> Dict[str, Any]:
    """
    Evaluates a single run against active scoring configuration.
    """
    case_res_objs: List[EvaluationResult] = []
    for _, er in run_evals.iterrows():
        ev_dict = {}
        try:
            ev_dict = json.loads(er.get("evidence_json", "{}") or "{}")
        except Exception:
            ev_dict = {}
        case_res_objs.append(EvaluationResult(
            metric_name=er["metric_name"],
            score=float(er["score"]),
            passed=bool(er["passed"]),
            threshold=float(er.get("threshold", 1.0) or 1.0),
            explanation=er.get("details", "") or er.get("explanation", ""),
            details=er.get("details", "") or er.get("explanation", ""),
            evaluator_type=er.get("evaluator_type", "deterministic"),
            evidence=ev_dict,
        ))

    summary = calculate_case_scores(case_res_objs, scoring_config)
    summary["eval_objects"] = case_res_objs
    return summary


def render_failure_analysis(
    test_case: Any,
    trace: Any,
    eval_results: List[Any],
    show_all: bool = False,
):
    """Render a structured Failure Analysis card for a failed evaluation.

    Uses FailureAnalyzer to classify the failure and display:
      - Failure type
      - Expected vs Actual
      - Evidence (observed, deterministic)
      - Likely Root Cause (inferred, clearly labelled)
      - Recommendation

    Args:
        test_case: TestCase object
        trace: Trace object with spans
        eval_results: List of EvaluationResult objects
        show_all: If True, show all applicable failure analyses, not just primary
    """
    from src.analysis.failure_analysis import FailureAnalyzer, FailureType

    has_failures = any(not getattr(r, "passed", True) for r in eval_results)
    if not has_failures:
        return

    if show_all:
        analyses = FailureAnalyzer.analyze_all(test_case, trace, eval_results)
    else:
        primary = FailureAnalyzer.analyze(test_case, trace, eval_results)
        if primary.failure_type == FailureType.NO_FAILURE:
            return
        analyses = [primary]

    if not analyses:
        return

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#fb7185;text-transform:uppercase;'
        'letter-spacing:0.1em;margin-top:16px;margin-bottom:8px;">🔬 FAILURE ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    for idx, fa in enumerate(analyses):
        # Failure type color
        type_colors = {
            FailureType.WRONG_TOOL: ("#a855f7", "🔀"),
            FailureType.INCORRECT_TOOL_ARGS: ("#f59e0b", "⚙️"),
            FailureType.INCORRECT_ANSWER: ("#f43f5e", "❌"),
            FailureType.POOR_GROUNDING: ("#e879f9", "📉"),
            FailureType.LATENCY_VIOLATION: ("#38bdf8", "⏱️"),
            FailureType.TOOL_API_FAILURE: ("#ef4444", "💥"),
        }
        color, icon = type_colors.get(fa.failure_type, ("#94a3b8", "🔹"))

        # Confidence label
        if fa.confidence >= 0.9:
            conf_label = "HIGH"
            conf_color = "#34d399"
        elif fa.confidence >= 0.7:
            conf_label = "MEDIUM"
            conf_color = "#fbbf24"
        else:
            conf_label = "LOW"
            conf_color = "#fb7185"

        # Main failure card
        card_html = (
            f'<div style="background:rgba(15,23,42,0.85);border:1px solid {color}44;border-radius:12px;'
            f'padding:18px 20px;margin-bottom:14px;box-shadow:0 4px 20px -2px rgba(0,0,0,0.4);position:relative;overflow:hidden;">'
            f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{color},transparent);"></div>'

            # Header: Failure label & type + confidence
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;">'
            f'<div>'
            f'<div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">FAILURE</div>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-size:18px;">{icon}</span>'
            f'<span style="font-size:16px;font-weight:800;color:#f8fafc;font-family:\'Plus Jakarta Sans\',sans-serif;">{fa.failure_type.value}</span>'
            f'</div>'
            f'</div>'
            f'<span style="background:{conf_color}1a;color:{conf_color};font-size:10.5px;font-weight:700;'
            f'padding:3px 10px;border-radius:9999px;border:1px solid {conf_color}33;'
            f'font-family:\'JetBrains Mono\',monospace;">'
            f'CONFIDENCE: {conf_label} ({fa.confidence:.0%})</span>'
            f'</div>'

            # Expected vs Actual
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'<div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.18);'
            f'border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:10px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">📋 EXPECTED</div>'
            f'<div style="font-size:12.5px;color:#e2e8f0;font-family:\'JetBrains Mono\',monospace;word-break:break-word;">{_escape_html(fa.expected)}</div>'
            f'</div>'
            f'<div style="background:rgba(244,63,94,0.06);border:1px solid rgba(244,63,94,0.18);'
            f'border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:10px;font-weight:700;color:#fb7185;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">💥 ACTUAL</div>'
            f'<div style="font-size:12.5px;color:#e2e8f0;font-family:\'JetBrains Mono\',monospace;word-break:break-word;">{_escape_html(fa.actual)}</div>'
            f'</div>'
            f'</div>'

            # Evidence (observed, deterministic fact)
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">'
            f'🔍 EVIDENCE <span style="font-weight:500;color:#64748b;font-size:9.5px;">(Observed Telemetry — Deterministic Fact)</span></div>'
            f'<div style="font-size:12px;color:#cbd5e1;font-family:\'JetBrains Mono\',monospace;line-height:1.4;">{_escape_html(fa.evidence)}</div>'
            f'</div>'

            # Likely Root Cause (inferred hypothesis — clearly separated and labeled)
            f'<div style="background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.16);'
            f'border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;color:#c084fc;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">'
            f'🧠 LIKELY ROOT CAUSE <span style="font-weight:500;color:#8b5cf6;font-size:9.5px;">(Inferred Diagnosis — Heuristic Hypothesis, Not Guaranteed)</span></div>'
            f'<div style="font-size:12.5px;color:#e4e4e7;line-height:1.45;">{_escape_html(fa.root_cause)}</div>'
            f'</div>'

            # Recommendation
            f'<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.16);'
            f'border-radius:8px;padding:10px 14px;">'
            f'<div style="font-size:10px;font-weight:700;color:#34d399;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">💡 RECOMMENDATION</div>'
            f'<div style="font-size:12.5px;color:#e4e4e7;line-height:1.45;">{_escape_html(fa.recommendation)}</div>'
            f'</div>'

            # Footer notice: explicit separation
            f'<div style="font-size:10px;color:#64748b;margin-top:10px;text-align:right;font-style:italic;">'
            f'⚠️ Telemetry evidence is deterministic. Root causes and recommendations are inferred diagnostic hypotheses.'
            f'</div>'

            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
        with st.expander("📋 Copy Plaintext Failure Summary", expanded=False):
            st.code(fa.format_text(include_disclaimer=True), language="yaml")


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for display in markdown cards."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

