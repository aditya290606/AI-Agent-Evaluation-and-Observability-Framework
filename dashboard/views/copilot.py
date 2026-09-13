"""
AI Evaluation Copilot View.

Provides an interactive analytical assistant that answers engineering questions
about agent failures, tool errors, regressions, latency bottlenecks, and costs,
grounded strictly in stored evaluation telemetry.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.analysis.copilot import EvaluationCopilot, CopilotResponse, CopilotFinding
from dashboard.views.common import render_section_header, render_kpi_card


STANDARD_QUESTIONS = [
    ("🚨 Why is my agent failing?", "Why is my agent failing?"),
    ("🛠️ Which tools cause the most failures?", "Which tools cause the most failures?"),
    ("📉 Which test cases regress most often?", "Which test cases regress most often?"),
    ("⚖️ Which agent version is better?", "Which agent version is better?"),
    ("⏱️ Why did latency increase?", "Why did latency increase?"),
    ("💰 Which model is most cost-efficient?", "Which model is most cost-efficient?"),
    ("📊 What are the most common failure categories?", "What are the most common failure categories?"),
    ("📚 Which RAG queries have poor retrieval?", "Which RAG queries have poor retrieval?"),
    ("🔄 Which tools are being called unnecessarily?", "Which tools are being called unnecessarily?"),
    ("🎯 What should I investigate first?", "What should I investigate first?"),
]


def render_copilot(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    all_runs: pd.DataFrame,
    all_evals: pd.DataFrame,
):
    render_section_header(
        title="AI Evaluation Copilot",
        subtitle="Ask natural language diagnostic questions about agent failures, tool bottlenecks, regressions, RAG grounding, and cost trade-offs. Grounded strictly in actual telemetry.",
        breadcrumb="INTELLIGENCE // AI COPILOT",
        action_badge="EVIDENCE GROUNDED",
    )

    # Initialize copilot session state
    if "copilot_query" not in st.session_state:
        st.session_state["copilot_query"] = "What should I investigate first?"
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []

    # Quick Scope Controls
    scope_c1, scope_c2, scope_c3 = st.columns(3)
    with scope_c1:
        agent_names = ["All Agents"] + sorted(list(set([r for r in all_runs["agent_name"].dropna().unique()]))) if not all_runs.empty else ["All Agents"]
        selected_agent = st.selectbox("Copilot Scope: Agent", agent_names, index=0, key="copilot_agent_scope")
    with scope_c2:
        version_names = ["All Versions"] + sorted(list(set([r for r in all_runs["agent_version"].dropna().unique()]))) if not all_runs.empty else ["All Versions"]
        selected_version = st.selectbox("Copilot Scope: Version", version_names, index=0, key="copilot_version_scope")
    with scope_c3:
        exp_names = ["All Datasets/Experiments"] + sorted(list(set([r for r in all_runs["experiment_id"].dropna().unique()]))) if not all_runs.empty else ["All Datasets/Experiments"]
        selected_exp = st.selectbox("Copilot Scope: Experiment", exp_names, index=0, key="copilot_exp_scope")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # QUICK-ASK QUESTION PILLS GRID (10 Standard Questions)
    # ---------------------------------------------------------------------------
    st.markdown(
        """ <div style="font-size: 11px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;"> ⚡ QUICK DIAGNOSTIC QUERIES (SELECT TO ASK) </div> """,
        unsafe_allow_html=True,
    )

    # Render 10 pills in 2 rows of 5
    row1_cols = st.columns(5)
    for idx, (label, q_text) in enumerate(STANDARD_QUESTIONS[:5]):
        with row1_cols[idx]:
            if st.button(label, key=f"q_pill_{idx}", width="stretch"):
                st.session_state["copilot_query"] = q_text

    row2_cols = st.columns(5)
    for idx, (label, q_text) in enumerate(STANDARD_QUESTIONS[5:]):
        with row2_cols[idx]:
            if st.button(label, key=f"q_pill_{idx+5}", width="stretch"):
                st.session_state["copilot_query"] = q_text

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # FREEFORM QUERY INPUT BAR
    # ---------------------------------------------------------------------------
    q_input_col, q_btn_col = st.columns([5, 1])
    with q_input_col:
        user_query = st.text_input(
            "Ask the AI Evaluation Copilot a question*",
            value=st.session_state.get("copilot_query", "What should I investigate first?"),
            placeholder="e.g., Why is my agent failing? Which tools cause the most failures?",
            key="input_copilot_query",
        )
    with q_btn_col:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        ask_btn = st.button("🚀 Ask Copilot", type="primary", width="stretch", key="btn_ask_copilot")

    active_query = user_query.strip() if user_query else "What should I investigate first?"

    # Execute Copilot Query
    with st.spinner("Analyzing stored evaluation telemetry and traces..."):
        response: CopilotResponse = EvaluationCopilot.ask(
            query=active_query,
            agent_name=selected_agent if selected_agent != "All Agents" else None,
            version=selected_version if selected_version != "All Versions" else None,
            experiment_id=selected_exp if selected_exp != "All Datasets/Experiments" else None,
        )

    # ---------------------------------------------------------------------------
    # COPILOT RESPONSE DISPLAY
    # ---------------------------------------------------------------------------
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Direct Answer Box
    border_accent = "#ef4444" if response.insufficient_data else "#38bdf8"
    box_bg = "rgba(15, 23, 42, 0.9)"

    st.markdown(
        f""" <div style=" background: {box_bg}; border: 1px solid {border_accent}55; border-left: 5px solid {border_accent}; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35); "> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"> <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8;"> 🤖 COPILOT DIRECT DIAGNOSIS </div> <div> <span style="font-size: 11px; color: #94a3b8; font-family: monospace;">Intent: {response.intent}</span> </div> </div> <div style="font-size: 16px; font-weight: 600; color: #f8fafc; line-height: 1.6;"> {response.direct_answer} </div> </div> """,
        unsafe_allow_html=True,
    )

    if response.insufficient_data:
        st.info("💡 **Tip:** Run evaluation benchmarks in the **📁 Datasets** or **📋 Reports** tabs to record telemetry for multi-dimensional copilot analysis.")
        return

    # ---------------------------------------------------------------------------
    # STRUCTURED FINDINGS (Finding / Evidence / Recommendation)
    # ---------------------------------------------------------------------------
    if response.findings:
        st.markdown(
            """ <div style="font-size: 13px; font-weight: 800; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;"> <span>📋</span> EVIDENCE-BACKED FINDINGS & RECOMMENDATIONS </div> """,
            unsafe_allow_html=True,
        )

        for idx, f in enumerate(response.findings):
            priority_color = "#ef4444" if "P0" in f.priority else ("#f59e0b" if "P1" in f.priority else "#38bdf8")
            
            st.markdown(
                f""" <div style=" background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 20px; margin-bottom: 16px; "> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 8px;"> <div style="font-size: 14px; font-weight: 700; color: #f8fafc;"> Finding #{idx+1}: {f.category} </div> <div style="display: flex; gap: 8px;"> <span style="background: rgba(255, 255, 255, 0.08); color: #cbd5e1; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{f.category}</span> <span style="background: {priority_color}22; color: {priority_color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid {priority_color}44;">{f.priority}</span> </div> </div> <div style="margin-bottom: 10px;"> <span style="font-size: 11px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px;">Finding:</span> <div style="color: #f1f5f9; font-size: 14px; font-weight: 600; margin-top: 2px;">{f.finding}</div> </div> <div style="margin-bottom: 12px; background: rgba(8, 12, 20, 0.6); padding: 10px 14px; border-radius: 6px; border-left: 3px solid #10b981;"> <span style="font-size: 11px; font-weight: 800; color: #10b981; text-transform: uppercase; letter-spacing: 0.5px;">Evidence:</span> <div style="color: #cbd5e1; font-size: 13.5px; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{f.evidence}</div> </div> <div> <span style="font-size: 11px; font-weight: 800; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.5px;">Recommendation:</span> <div style="color: #e2e8f0; font-size: 13.5px; margin-top: 2px;">💡 {f.recommendation}</div> </div> </div> """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------------------
    # SUPPORTING DATASET TABLE
    # ---------------------------------------------------------------------------
    if response.supporting_table:
        st.markdown(
            """ <div style="font-size: 13px; font-weight: 800; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 20px; margin-bottom: 10px;"> 📊 Supporting Telemetry Data </div> """,
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(response.supporting_table), width="stretch", hide_index=True)

    # ---------------------------------------------------------------------------
    # ONE-CLICK DRILL-DOWN SHORTCUTS
    # ---------------------------------------------------------------------------
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """ <div style="font-size: 11px; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;"> 🔗 RELEVANT INVESTIGATION SHORTCUTS </div> """,
        unsafe_allow_html=True,
    )
    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        if st.button("🚨 Inspect Failures & RCA", width="stretch", key="btn_drill_rca"):
            st.session_state["nav_section"] = "🚨 Failures"
            st.rerun()
    with s_c2:
        if st.button("🔍 View Trace Spans & Gantt", width="stretch", key="btn_drill_traces"):
            st.session_state["nav_section"] = "🔍 Traces"
            st.rerun()
    with s_c3:
        if st.button("🛠️ Tool Analytics Breakdown", width="stretch", key="btn_drill_tools"):
            st.session_state["nav_section"] = "🛠️ Tool Analytics"
            st.rerun()
    with s_c4:
        if st.button("📋 Open Reports & Audits", width="stretch", key="btn_drill_reports"):
            st.session_state["nav_section"] = "📋 Reports"
            st.rerun()
