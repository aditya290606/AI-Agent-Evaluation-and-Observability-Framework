"""
Agent Evaluation & Observability Console - Interactive Platform.

Redesigned 12-section professional console with global filtering,
health indicators, deep observability, and seamless drill-down navigation.
"""

import json
import os
import sys
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, AgentRecord
from src.registry.registry import AgentRegistry
from src.core.test_case_manager import TestCaseManager
from src.core.dataset_manager import DatasetManager
from src.evaluation.batch_runner import BatchEvaluationRunner
from src.sandbox import SandboxManager
from src.evaluation.scoring import ScoringConfig
from src.cost import global_cost_calculator

# View Modules
from dashboard.views.overview import render_overview
from dashboard.views.agents import render_agents
from dashboard.views.datasets import render_datasets
from dashboard.views.test_cases import render_test_cases
from dashboard.views.eval_runs import render_eval_runs
from dashboard.views.traces import render_traces
from dashboard.views.metrics import render_metrics
from dashboard.views.failures import render_failures
from dashboard.views.experiments import render_experiments
from dashboard.views.rag_analytics import render_rag_analytics
from dashboard.views.tool_analytics import render_tool_analytics
from dashboard.views.cost_performance import render_cost_performance
from dashboard.views.reports import render_reports
from dashboard.views.copilot import render_copilot
from dashboard.views.connect_agent import render_connect_agent

# Page Configuration
st.set_page_config(
    page_title="Agent Eval & Observability Console",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# Initialize Database
init_db()

# Luxury Dark Aesthetic Design System CSS
st.markdown(
    """ <style> @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap'); /* Hide Streamlit default header chrome */ header[data-testid="stHeader"] { display: none !important; } #MainMenu { visibility: hidden !important; } footer { visibility: hidden !important; } /* Global Reset & Background */ html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; } .stApp { background: radial-gradient(ellipse at 50% -10%, #0d1a33 0%, #080c14 70%, #05080e 100%) fixed !important; color: #f1f5f9; } /* Subtle scrollbars */ ::-webkit-scrollbar { width: 6px; height: 6px; } ::-webkit-scrollbar-track { background: rgba(8, 12, 20, 0.5); } ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.2); border-radius: 4px; } ::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.4); } /* Sidebar Styling */ section[data-testid="stSidebar"] { background: rgba(10, 15, 29, 0.95) !important; border-right: 1px solid rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) !important; min-width: 290px !important; } section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem; } /* Sidebar Section Group Headers */ .sidebar-group-header { font-size: 10px !important; font-weight: 700 !important; letter-spacing: 0.12em !important; color: #64748b !important; text-transform: uppercase !important; margin-top: 14px !important; margin-bottom: 6px !important; padding-left: 6px !important; } /* Sidebar Nav Buttons Styling */ section[data-testid="stSidebar"] div[data-testid="stButton"] { margin-bottom: 3px !important; } section[data-testid="stSidebar"] div[data-testid="stButton"] > button { display: flex !important; justify-content: flex-start !important; align-items: center !important; text-align: left !important; width: 100% !important; border-radius: 8px !important; padding: 8px 12px !important; font-size: 12.5px !important; font-weight: 600 !important; letter-spacing: 0.01em !important; transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important; box-shadow: none !important; min-height: 38px !important; } /* Inactive Nav Buttons (Secondary) */ section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] { background: rgba(15, 23, 42, 0.45) !important; color: #94a3b8 !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; } section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover { background: rgba(56, 189, 248, 0.1) !important; color: #f1f5f9 !important; border-color: rgba(56, 189, 248, 0.35) !important; transform: translateX(4px) !important; } /* Active Nav Button (Primary) */ section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(90deg, rgba(56, 189, 248, 0.22) 0%, rgba(99, 102, 241, 0.16) 100%) !important; color: #38bdf8 !important; font-weight: 700 !important; border: 1px solid rgba(56, 189, 248, 0.5) !important; border-left: 4px solid #38bdf8 !important; box-shadow: 0 0 16px rgba(56, 189, 248, 0.2), inset 0 0 8px rgba(56, 189, 248, 0.06) !important; transform: translateX(2px) !important; } /* Segmented Control Styling */ div[data-testid="stSegmentedControl"] { background: rgba(15, 23, 42, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 8px !important; padding: 3px !important; } div[data-testid="stSegmentedControl"] button { border-radius: 6px !important; font-size: 11.5px !important; font-weight: 600 !important; color: #94a3b8 !important; transition: all 0.15s ease !important; } div[data-testid="stSegmentedControl"] button[aria-checked="true"] { background: rgba(56, 189, 248, 0.22) !important; color: #38bdf8 !important; border: 1px solid #38bdf8 !important; font-weight: 700 !important; } /* Sidebar Filter Labels */ section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p, section[data-testid="stSidebar"] .stSelectbox label p, section[data-testid="stSidebar"] .stMultiSelect label p { color: #94a3b8 !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; margin-bottom: 2px !important; display: block !important; visibility: visible !important; } /* BaseWeb Selectboxes & MultiSelect */ div[data-baseweb="select"] { border-radius: 8px !important; } div[data-baseweb="select"] > div { background-color: rgba(15, 23, 42, 0.75) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 8px !important; min-height: 38px !important; } div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #f1f5f9 !important; font-size: 12.5px !important; font-weight: 500 !important; } div[data-baseweb="select"]:hover > div { border-color: rgba(56, 189, 248, 0.4) !important; } div[data-baseweb="popover"] { background: #0b1120 !important; border: 1px solid rgba(255, 255, 255, 0.12) !important; border-radius: 8px !important; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important; } div[data-baseweb="popover"] li { color: #cbd5e1 !important; font-size: 12.5px !important; } div[data-baseweb="popover"] li:hover { background-color: rgba(56, 189, 248, 0.15) !important; color: #38bdf8 !important; } /* Glassmorphic KPI Cards Hover Lift */ .kpi-glass-card { transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important; } .kpi-glass-card:hover { transform: translateY(-3px) !important; box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(56, 189, 248, 0.3) !important; } /* Run Row Container Card */ .run-row-card { background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; transition: all 0.2s ease; } .run-row-card:hover { background: rgba(15, 23, 42, 0.85); border-color: rgba(56, 189, 248, 0.25); transform: translateY(-1px); } /* Buttons Styling */ .stButton > button { background: rgba(30, 41, 59, 0.8) !important; color: #f8fafc !important; border: 1px solid rgba(255, 255, 255, 0.12) !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 12.5px !important; padding: 6px 14px !important; transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important; backdrop-filter: blur(8px) !important; } .stButton > button:hover { border-color: #38bdf8 !important; color: #38bdf8 !important; box-shadow: 0 0 14px rgba(56, 189, 248, 0.25) !important; transform: translateY(-1px) !important; } .stButton > button[kind="primary"] { background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important; color: #ffffff !important; border: none !important; box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important; } .stButton > button[kind="primary"]:hover { box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important; transform: translateY(-1px) !important; } /* Streamlit Expanders */ div[data-testid="stExpander"] { background: rgba(15, 23, 42, 0.5) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 10px !important; margin-bottom: 12px !important; } /* Streamlit Tabs */ button[data-baseweb="tab"] { font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600 !important; font-size: 13px !important; color: #94a3b8 !important; padding: 10px 16px !important; border-bottom: 2px solid transparent !important; } button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; text-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important; } input, textarea { background-color: rgba(15, 23, 42, 0.8) !important; border-color: rgba(255, 255, 255, 0.1) !important; color: #f1f5f9 !important; border-radius: 8px !important; } /* Streamlit Metric Overrides */ div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; } </style> """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cached resource singletons — instantiated once, shared across all reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_registry():
    return AgentRegistry()

@st.cache_resource
def _get_tc_manager():
    return TestCaseManager()

@st.cache_resource
def _get_ds_manager():
    tc = _get_tc_manager()
    return DatasetManager(tc_manager=tc)

@st.cache_resource
def _get_batch_runner():
    return BatchEvaluationRunner()

@st.cache_resource
def _get_sandbox_manager():
    return SandboxManager()


# ---------------------------------------------------------------------------
# Data Loading — cached for 60 s; uses joinedload to eliminate N+1 DB queries
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_data():
    """Load all runs + eval results in two optimised queries (no N+1)."""
    from sqlalchemy.orm import joinedload
    session = get_session()
    try:
        # Single query: all runs with their eval_results eagerly joined
        runs = (
            session.query(Run)
            .options(joinedload(Run.eval_results))
            .order_by(Run.created_at.desc())
            .all()
        )

        # Build agent_id -> integration_type lookup
        agent_records = session.query(AgentRecord).all()
        agent_integration_map = {a.agent_id: (a.integration_type or "mock") for a in agent_records}

        run_rows, eval_rows = [], []
        for r in runs:
            model_name = getattr(r, "model", None) or "claude-3-5-haiku"
            pricing = global_cost_calculator.get_model_pricing(model_name)
            in_tok = r.total_input_tokens or 0
            out_tok = r.total_output_tokens or 0
            est_cost = pricing.calculate_cost(in_tok, out_tok)
            actual_c = getattr(r, "actual_cost_usd", None)

            # Compute run-level quality score and pass status
            scores = [ev.score for ev in r.eval_results if ev.score is not None]
            avg_score = float(sum(scores) / len(scores) * 100.0) if scores else 90.0
            all_passed = all(bool(ev.passed) for ev in r.eval_results) if r.eval_results else True

            # Determine integration type from agent registry
            run_agent_id = getattr(r, "agent_id", None) or "demo"
            integration_type = agent_integration_map.get(run_agent_id, "mock" if r.is_mock else "live")

            run_rows.append({
                "run_id": r.id,
                "experiment_id": getattr(r, "experiment_id", None) or "legacy",
                "agent_id": run_agent_id,
                "agent_name": getattr(r, "agent_name", None) or "DemoAgent",
                "agent_version": getattr(r, "agent_version", None) or "v1.0",
                "model": model_name,
                "provider": pricing.provider,
                "input_token_price": pricing.input_price_per_mtoken,
                "output_token_price": pricing.output_price_per_mtoken,
                "prompt_version": getattr(r, "prompt_version", None) or "v1.0",
                "dataset_version": getattr(r, "dataset_version", None) or "v1.0",
                "environment": getattr(r, "environment", None) or "production",
                "task_id": r.task_id,
                "trace_id": getattr(r, "trace_id", None) or r.task_id,
                "query": r.query,
                "final_answer": r.final_answer,
                "expected_tool": r.expected_tool,
                "tools_called": r.tools_called,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "est_cost_usd": est_cost,
                "actual_cost_usd": actual_c,
                "is_estimated": actual_c is None,
                "cost_type": "ACTUAL_PROVIDER" if actual_c is not None else "ESTIMATED",
                "latency_ms": r.latency_ms,
                "is_mock": r.is_mock,
                "integration_type": integration_type,
                "created_at": r.created_at,
                "score": avg_score / 100.0,
                "quality_score": avg_score,
                "passed": all_passed,
            })
            for ev in r.eval_results:
                eval_rows.append({
                    "run_id": r.id,
                    "task_id": r.task_id,
                    "created_at": r.created_at,
                    "agent_id": run_agent_id,
                    "agent_version": getattr(r, "agent_version", None) or "v1.0",
                    "metric_name": ev.metric_name,
                    "score": ev.score,
                    "passed": ev.passed,
                    "threshold": getattr(ev, "threshold", 1.0) or 1.0,
                    "evaluator_type": getattr(ev, "evaluator_type", "deterministic") or "deterministic",
                    "details": getattr(ev, "details", None) or getattr(ev, "explanation", None) or "",
                    "explanation": getattr(ev, "details", None) or getattr(ev, "explanation", None) or "",
                    "evidence_json": getattr(ev, "evidence_json", None) or "{}",
                })
        return pd.DataFrame(run_rows), pd.DataFrame(eval_rows)
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_steps(run_id: Any):
    session = get_session()
    try:
        steps = session.query(Step).filter(Step.run_id == run_id).order_by(Step.step_index).all()
        return steps
    finally:
        session.close()


# Resolve cached singletons
registry = _get_registry()
tc_manager = _get_tc_manager()
ds_manager = _get_ds_manager()
batch_runner = _get_batch_runner()
sandbox_manager = _get_sandbox_manager()

@st.cache_data(ttl=60)
def _load_tags():
    return _get_tc_manager().get_all_tags()

all_tags = _load_tags()

# Load data (served from cache until 60 s TTL expires)
runs_df, evals_df = load_data()


# ---------------------------------------------------------------------------
# Sidebar: Brand, Primary Navigation & Global Filters
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """ <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px; padding: 6px 4px;"> <div style="width: 40px; height: 40px; border-radius: 10px; background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(56, 189, 248, 0.45);"> <span style="font-size: 22px;">⚡</span> </div> <div> <div style="font-size: 16px; font-weight: 800; color: #f8fafc; letter-spacing: 0.03em;">AgentPulse</div> <div style="font-size: 10px; color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: 0.08em; display: flex; align-items: center; gap: 4px;"> <span style="width: 5px; height: 5px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981; display: inline-block;"></span> EVAL CONSOLE · LIVE </div> </div> </div> <div style="font-size: 10px; color: #334155; font-family: 'JetBrains Mono', monospace; margin-bottom: 16px; padding-left: 4px;">hackathon demo build</div> """,
    unsafe_allow_html=True,
)

# ── Demo flow sections (prominently pinned at top) ──────────────────────────
DEMO_SECTIONS = [
    ("📊 Overview",              "overview"),
    ("🚀 Evaluation Runs",       "eval_runs"),
    ("🚨 Failures",              "failures"),
    ("🔍 Traces",                "traces"),
    ("⚖️ Experiments / Versions", "experiments"),
]

# ── Full platform sections (collapsed under Advanced) ───────────────────────
ADVANCED_GROUPS = [
    ("PLATFORM", [
        ("⚡ Connect Agent",      "connect_agent"),
        ("🤖 Agents",             "agents"),
        ("📁 Datasets",           "datasets"),
        ("🧪 Test Cases",         "test_cases"),
    ]),
    ("INTELLIGENCE", [
        ("🤖 AI Copilot",         "copilot"),
        ("📈 Metrics",            "metrics"),
        ("📚 RAG Analytics",      "rag_analytics"),
        ("🛠️ Tool Analytics",     "tool_analytics"),
    ]),
    ("FINANCIALS", [
        ("💰 Cost & Performance", "cost_performance"),
        ("📋 Reports",            "reports"),
    ]),
]

# Flat list of every navigable section for state tracking
NAV_GROUPS = [("AGENTPULSE DEMO", DEMO_SECTIONS)] + ADVANCED_GROUPS

ALL_SECTIONS = [label for _, items in NAV_GROUPS for label, _ in items]

# Sync nav section with session state
if "nav_section" not in st.session_state or st.session_state["nav_section"] not in ALL_SECTIONS:
    st.session_state["nav_section"] = ALL_SECTIONS[0]

# ── Demo section buttons (always visible, top priority) ─────────────────────
st.sidebar.markdown(
    '<div style="font-size:10px; font-weight:700; color:#38bdf8; text-transform:uppercase; '
    'letter-spacing:.12em; margin-bottom:8px; padding: 0 2px;">🎬 DEMO FLOW</div>',
    unsafe_allow_html=True,
)
for label, code in DEMO_SECTIONS:
    is_active = (st.session_state["nav_section"] == label)
    if st.sidebar.button(
        label, key=f"nav_btn_{code}", width="stretch",
        type="primary" if is_active else "secondary"
    ):
        if not is_active:
            st.session_state["nav_section"] = label
            st.rerun()

# ── Advanced sections (collapsed by default) ─────────────────────────────────
st.sidebar.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
with st.sidebar.expander("⚙️ Advanced Sections", expanded=False):
    for group_title, items in ADVANCED_GROUPS:
        st.markdown(
            f'<div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase; '
            f'letter-spacing:.1em; margin: 8px 0 4px 0;">{group_title}</div>',
            unsafe_allow_html=True,
        )
        for label, code in items:
            is_active = (st.session_state["nav_section"] == label)
            if st.button(
                label, key=f"nav_btn_{code}", width="stretch",
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    st.session_state["nav_section"] = label
                    st.rerun()

selected_section = st.session_state["nav_section"]

st.sidebar.markdown("<div style='margin-top: 14px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

# Refresh Data button
_refresh_col, _ts_col = st.sidebar.columns([1, 1])
with _refresh_col:
    if st.button("🔄 Refresh Data", key="sidebar_refresh_btn", width="stretch", help="Reload run data from the database"):
        load_data.clear()
        load_steps.clear()
        _load_tags.clear()
        st.rerun()
with _ts_col:
    import datetime as _dt
    st.markdown(
        f"<div style='font-size:10px; color:#64748b; padding-top:10px; text-align:right;'>"
        f"Cache: 60 s<br/>Live on nav</div>",
        unsafe_allow_html=True,
    )

# Global Filters Header
st.sidebar.markdown(
    """ <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;"> <span>🔍</span> GLOBAL FILTERS </div> """,
    unsafe_allow_html=True,
)


# Filter: Agent
all_agents = ["All Agents"] + sorted(runs_df["agent_name"].dropna().unique().tolist()) if not runs_df.empty else ["All Agents"]
selected_agent = st.sidebar.selectbox("Agent", all_agents, index=0)

# Filter: Agent Version (dependent)
if selected_agent != "All Agents" and not runs_df.empty:
    agent_runs = runs_df[runs_df["agent_name"] == selected_agent]
    avail_versions = ["All Versions"] + sorted(agent_runs["agent_version"].dropna().unique().tolist())
else:
    avail_versions = ["All Versions"] + (sorted(runs_df["agent_version"].dropna().unique().tolist()) if not runs_df.empty else [])
selected_version = st.sidebar.selectbox("Agent Version", avail_versions, index=0)

# Filter: Date Range
with st.sidebar.expander("📅 Date Range Filter", expanded=False):
    date_filter_active = st.checkbox("Enable Date Range Filter", value=False)
    start_date = st.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=30))
    end_date = st.date_input("End Date", value=datetime.date.today())

# Filter: Dataset / Experiment
all_exp = ["All Datasets/Experiments"] + (sorted(runs_df["experiment_id"].dropna().unique().tolist()) if not runs_df.empty else [])
selected_dataset = st.sidebar.selectbox("Dataset / Experiment", all_exp, index=0)

# Filter: Test ID
all_task_ids = sorted(runs_df["task_id"].dropna().unique().tolist()) if not runs_df.empty else []
selected_tests = st.sidebar.multiselect("Test ID", all_task_ids, default=[], help="Leave empty to include all test IDs.")

# Filter: Execution Mode
mode_choice = st.sidebar.segmented_control(
    "Execution Mode",
    ["All", "Mock Only", "Live Only", "SDK Only"],
    default="All",
    key="sb_exec_mode"
)
mode_filter = mode_choice if mode_choice is not None else "All"

# Filter: Model
all_models = ["All Models"] + (sorted(runs_df["model"].dropna().unique().tolist()) if not runs_df.empty else [])
selected_model = st.sidebar.selectbox("Model", all_models, index=0)

# Filter: Metric
all_metric_names = sorted(evals_df["metric_name"].dropna().unique().tolist()) if not evals_df.empty else []
preselected_metrics = st.session_state.pop("global_metric_filter", [])
selected_metrics = st.sidebar.multiselect(
    "Evaluation Metric",
    all_metric_names,
    default=[m for m in preselected_metrics if m in all_metric_names],
    help="Filter evaluation assertions."
)

# Filter: Status (All / Passed / Failed)
status_choice = st.sidebar.segmented_control(
    "Status",
    ["All", "Passed", "Failed"],
    default="All",
    key="sb_status"
)
status_filter = status_choice if status_choice is not None else "All"

# Sidebar: Scoring Weights & Presets
with st.sidebar.expander("⚖️ Scoring Presets & Weights", expanded=False):
    weight_preset = st.selectbox(
        "Scoring Preset",
        [
            "Balanced (30% Task, 20% Tool, 20% Grounded, 15% Quality, 10% Latency, 5% Cost)",
            "Quality & Groundedness First",
            "Latency & Cost Optimized",
            "Custom Weights",
        ],
        index=0,
    )

    if "Quality" in weight_preset:
        active_scoring_config = ScoringConfig.quality_first_preset()
    elif "Latency" in weight_preset:
        active_scoring_config = ScoringConfig.efficiency_preset()
    elif "Custom" in weight_preset:
        w_task = st.slider("Task Success Weight", 0.0, 1.0, 0.30, 0.05)
        w_tool = st.slider("Tool Accuracy Weight", 0.0, 1.0, 0.20, 0.05)
        w_ground = st.slider("Groundedness Weight", 0.0, 1.0, 0.20, 0.05)
        w_ans = st.slider("Answer Quality Weight", 0.0, 1.0, 0.15, 0.05)
        w_lat = st.slider("Latency Budget Weight", 0.0, 1.0, 0.10, 0.05)
        w_cost = st.slider("Cost Budget Weight", 0.0, 1.0, 0.05, 0.05)
        active_scoring_config = ScoringConfig(
            weights={
                "task_success": w_task,
                "tool_accuracy": w_tool,
                "groundedness": w_ground,
                "answer_quality": w_ans,
                "latency_budget": w_lat,
                "cost_budget": w_cost,
            }
        )
    else:
        active_scoring_config = ScoringConfig.balanced_preset()


# ---------------------------------------------------------------------------
# Apply Global Filters to Data
# ---------------------------------------------------------------------------

filtered_runs = runs_df.copy() if not runs_df.empty else pd.DataFrame()
filtered_evals = evals_df.copy() if not evals_df.empty else pd.DataFrame()

if not filtered_runs.empty:
    if selected_agent != "All Agents":
        filtered_runs = filtered_runs[filtered_runs["agent_name"] == selected_agent]
    if selected_version != "All Versions":
        filtered_runs = filtered_runs[filtered_runs["agent_version"] == selected_version]
    if selected_dataset != "All Datasets/Experiments":
        filtered_runs = filtered_runs[filtered_runs["experiment_id"] == selected_dataset]
    if selected_tests:
        filtered_runs = filtered_runs[filtered_runs["task_id"].isin(selected_tests)]
    if selected_model != "All Models":
        filtered_runs = filtered_runs[filtered_runs["model"] == selected_model]
    if mode_filter == "Mock Only":
        filtered_runs = filtered_runs[filtered_runs["is_mock"] == True]
    elif mode_filter == "Live Only":
        filtered_runs = filtered_runs[(filtered_runs["is_mock"] == False) & (filtered_runs["integration_type"] != "sdk")]
    elif mode_filter == "SDK Only":
        filtered_runs = filtered_runs[filtered_runs["integration_type"] == "sdk"]

    if date_filter_active and "created_at" in filtered_runs.columns:
        filtered_runs["created_date"] = pd.to_datetime(filtered_runs["created_at"]).dt.date
        filtered_runs = filtered_runs[(filtered_runs["created_date"] >= start_date) & (filtered_runs["created_date"] <= end_date)]

    # Filter evals by matching run_ids
    if not filtered_evals.empty:
        filtered_evals = filtered_evals[filtered_evals["run_id"].isin(filtered_runs["run_id"])]
        if selected_metrics:
            filtered_evals = filtered_evals[filtered_evals["metric_name"].isin(selected_metrics)]

    # Status filter
    if status_filter != "All" and not filtered_evals.empty:
        if status_filter == "Passed":
            passed_run_ids = filtered_evals.groupby("run_id")["passed"].all()
            valid_ids = passed_run_ids[passed_run_ids == True].index.tolist()
            filtered_runs = filtered_runs[filtered_runs["run_id"].isin(valid_ids)]
            filtered_evals = filtered_evals[filtered_evals["run_id"].isin(valid_ids)]
        elif status_filter == "Failed":
            failed_run_ids = filtered_evals[filtered_evals["passed"] == False]["run_id"].unique().tolist()
            filtered_runs = filtered_runs[filtered_runs["run_id"].isin(failed_run_ids)]
            filtered_evals = filtered_evals[filtered_evals["run_id"].isin(failed_run_ids)]


# ---------------------------------------------------------------------------
# Section Dispatcher
# ---------------------------------------------------------------------------

if selected_section == "📊 Overview":
    render_overview(filtered_runs, filtered_evals, runs_df, evals_df, active_scoring_config, tc_manager)

elif selected_section == "⚡ Connect Agent":
    render_connect_agent()

elif selected_section == "🤖 Agents":
    render_agents(registry, sandbox_manager)

elif selected_section == "📁 Datasets":
    render_datasets(ds_manager, tc_manager, batch_runner, registry, active_scoring_config)

elif selected_section == "🧪 Test Cases":
    render_test_cases(tc_manager, registry)

elif selected_section == "🚀 Evaluation Runs":
    render_eval_runs(filtered_runs, filtered_evals, active_scoring_config, load_steps_fn=load_steps, tc_manager=tc_manager)

elif selected_section == "🔍 Traces":
    render_traces(filtered_runs, filtered_evals, load_steps, active_scoring_config, tc_manager=tc_manager)

elif selected_section == "📈 Metrics":
    render_metrics(filtered_runs, filtered_evals)

elif selected_section == "🚨 Failures":
    render_failures(filtered_runs, filtered_evals, load_steps, tc_manager, active_scoring_config)

elif selected_section == "🤖 AI Copilot":
    render_copilot(filtered_runs, filtered_evals, runs_df, evals_df)

elif selected_section == "⚖️ Experiments / Versions":
    render_experiments(runs_df, evals_df)

elif selected_section == "📚 RAG Analytics":
    render_rag_analytics(filtered_runs, filtered_evals, load_steps, tc_manager)

elif selected_section == "🛠️ Tool Analytics":
    render_tool_analytics(filtered_runs, filtered_evals, load_steps, tc_manager)

elif selected_section == "💰 Cost & Performance":
    render_cost_performance(filtered_runs, filtered_evals)

elif selected_section == "📋 Reports":
    render_reports(filtered_runs, filtered_evals, runs_df, evals_df, registry, active_scoring_config, all_tags)
