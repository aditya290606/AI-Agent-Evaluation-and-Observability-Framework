"""
Section 11: Cost & Performance - Cost Observability, Model Breakdown, Pareto Frontier, and Pricing Catalogs.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from src.cost import (
    CostAnalyticsEngine,
    DEFAULT_MODEL_PRICING,
    DEFAULT_TOOL_PRICING,
)
from dashboard.views.common import render_kpi_card, render_section_header, apply_plotly_theme


def render_cost_performance(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
):
    render_section_header(
        title="Cost & Performance Observability",
        subtitle="Model-aware token pricing, tool API tracking, cost-vs-quality Pareto trade-offs, and efficiency metrics.",
        breadcrumb="OBSERVABILITY // COST",
        action_badge="COST TRACKER",
    )

    # 1. Cost Integrity Disclaimer Banner
    st.markdown(
        """ <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;"> <div style="font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em;">ℹ️ Cost Integrity & Provider Accounting Policy:</div> <div style="margin-top: 4px; font-size: 13px; color: #cbd5e1;"> All LLM and tool execution costs are derived from verified model-specific pricing catalogs and actual token counts (<strong style="color: #93c5fd;">📊 Estimated Cost</strong>). Exact upstream provider billing headers are preserved and distinguished as <strong style="color: #34d399;">🔒 Actual Provider Cost</strong>. <em>Estimated costs are transparently differentiated from exact invoice billing.</em> </div> </div> """,
        unsafe_allow_html=True,
    )

    if filtered_runs.empty:
        st.info("No evaluation runs logged yet. Execute an evaluation to view cost observability data.")
        return

    # Cost Summary KPIs
    cost_summary = CostAnalyticsEngine.compute_cost_summary(filtered_runs)

    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
    with kpi_c1:
        render_kpi_card("Total Est. Cost", f"${cost_summary['total_estimated_cost_usd']:.4f}", subtitle="Catalog Model + Tools", icon="🪙", accent_color="#38bdf8")
    with kpi_c2:
        act_disp = f"${cost_summary['total_actual_cost_usd']:.4f}" if cost_summary['has_actual_cost'] else "Simulated"
        act_status = "🔒 Actual Provider" if cost_summary['has_actual_cost'] else "⚪ Estimated"
        render_kpi_card("Actual Provider Cost", act_disp, subtitle=act_status, icon="🔒", accent_color="#10b981" if cost_summary['has_actual_cost'] else "#64748b")
    with kpi_c3:
        render_kpi_card("Avg Cost / Run", f"${cost_summary['cost_per_run_avg']:.5f}", subtitle="Mean Task Cost", icon="📊", accent_color="#818cf8")
    with kpi_c4:
        render_kpi_card("Cost / Success", f"${cost_summary['cost_per_successful_task']:.5f}", subtitle="Per Successful Task", icon="💎", accent_color="#06b6d4")
    with kpi_c5:
        render_kpi_card("Cost Efficiency", f"{cost_summary['avg_cost_efficiency']:,.0f}", subtitle="Quality pts / $ Spent", icon="⚡", accent_color="#f59e0b")
    with kpi_c6:
        render_kpi_card("Total Tokens", f"{cost_summary['total_tokens']:,}", subtitle="Cumulative Input/Output", icon="🔢", accent_color="#a855f7")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Section 2: Visual Cost Analytics Charts
    st.markdown("### 📊 Cost Breakdown by Model & Agent")

    c_chart1, c_chart2 = st.columns(2)

    # 1. Cost by Model & Provider
    with c_chart1:
        st.markdown("#### 🤖 Cost by Model & Provider")
        model_cost_df = CostAnalyticsEngine.compute_cost_by_model(filtered_runs)
        if not model_cost_df.empty:
            fig_m = px.bar(
                model_cost_df,
                x="model",
                y="total_cost_usd",
                color="model",
                text_auto=".4f",
                labels={"model": "LLM Model", "total_cost_usd": "Total Cost ($ USD)"},
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f59e0b"],
            )
            fig_m.update_traces(marker_line_width=0, opacity=0.9)
            apply_plotly_theme(fig_m, height=300)
            st.plotly_chart(fig_m, width="stretch")

            st.dataframe(
                model_cost_df[["model", "total_runs", "total_tokens", "total_cost_usd", "avg_cost_per_run", "cost_efficiency"]],
                width="stretch"
            )

    # 2. Cost by Agent
    with c_chart2:
        st.markdown("#### 👥 Cost by Agent & Version")
        agent_cost_df = CostAnalyticsEngine.compute_cost_by_agent(filtered_runs)
        if not agent_cost_df.empty:
            fig_a = px.bar(
                agent_cost_df,
                x="agent",
                y="total_cost_usd",
                color="version",
                barmode="group",
                text_auto=".4f",
                labels={"agent": "Agent", "total_cost_usd": "Total Cost ($ USD)"},
                color_discrete_sequence=["#34d399", "#38bdf8", "#a855f7"],
            )
            fig_a.update_traces(marker_line_width=0, opacity=0.9)
            apply_plotly_theme(fig_a, height=300)
            st.plotly_chart(fig_a, width="stretch")

            st.dataframe(
                agent_cost_df[["agent", "version", "total_runs", "total_cost_usd", "avg_cost_per_run", "cost_efficiency"]],
                width="stretch"
            )

    st.markdown("<div style='margin-top: 14px; margin-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    # Section 3: Cost vs Quality Trade-off (Pareto Frontier)
    st.markdown("### ⚖️ Cost vs. Quality Trade-Off Analysis")
    st.caption("Identify models and runs that deliver maximum accuracy at minimal cost (high cost-efficiency quadrant).")

    cost_vs_qual_df = CostAnalyticsEngine.compute_cost_vs_quality(filtered_runs, filtered_evals)
    if not cost_vs_qual_df.empty:
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            fig_scatter = px.scatter(
                cost_vs_qual_df,
                x="cost_usd",
                y="quality_score",
                color="model",
                hover_data=["task_id", "latency_ms", "cost_efficiency"],
                labels={"cost_usd": "Estimated Cost ($ USD)", "quality_score": "Quality Score (%)", "model": "Model"},
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f59e0b"],
            )
            fig_scatter.add_hline(y=80, line_dash="dash", line_color="#10b981", annotation_text="Quality Gate (80%)")
            apply_plotly_theme(fig_scatter, height=360)
            st.plotly_chart(fig_scatter, width="stretch")
        with q_col2:
            st.markdown("##### 💡 Quadrant Breakdown")
            st.markdown(
                """ - 🌟 **Optimal (Top-Left)**: High Quality (>80%), Low Cost - ⚠️ **Costly (Top-Right)**: High Quality (>80%), High Cost - 🔴 **Suboptimal (Bottom-Right)**: Low Quality (<80%), High Cost - ⚪ **Budget (Bottom-Left)**: Low Quality (<80%), Low Cost """
            )
            top_efficient = cost_vs_qual_df.sort_values(by="cost_efficiency", ascending=False).iloc[0] if not cost_vs_qual_df.empty else None
            if top_efficient is not None:
                st.success(f"🏆 **Most Efficient Run:** `{top_efficient['task_id']}` ({top_efficient['cost_efficiency']:,.0f} pts/$) using `{top_efficient['model']}`")

    st.markdown("<div style='margin-top: 14px; margin-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    # Section 4: Model Pricing Catalog Inspector & Custom Simulator
    st.markdown("### 🏷️ Model & Tool Pricing Configuration")
    st.caption("Inspect verified per-1M token rates across providers, or simulate custom enterprise rates.")

    catalog_tab1, catalog_tab2, catalog_tab3 = st.tabs([
        "📋 LLM Pricing Catalog",
        "🛠️ Tool / API Pricing",
        "🧮 Live Rate Simulator & Custom Override",
    ])

    with catalog_tab1:
        pricing_records = []
        for m_id, p in DEFAULT_MODEL_PRICING.items():
            pricing_records.append({
                "Model": p.display_name,
                "Provider": p.provider.upper(),
                "Input $/1M Tokens": f"${p.input_price_per_mtoken:.2f}",
                "Output $/1M Tokens": f"${p.output_price_per_mtoken:.2f}",
                "Prompt Cache $/1M Tokens": f"${p.cached_input_price_per_mtoken:.4f}" if p.cached_input_price_per_mtoken else "—",
                "Context Window": f"{p.context_window:,} tokens",
            })
        st.dataframe(pd.DataFrame(pricing_records), width="stretch")

    with catalog_tab2:
        tool_pricing_records = [
            {"Tool / API": name, "Cost per Invocation ($ USD)": f"${cost:.4f}", "Pricing Model": "Per Execution Call" if cost > 0 else "Free / Internal"}
            for name, cost in DEFAULT_TOOL_PRICING.items()
        ]
        st.dataframe(pd.DataFrame(tool_pricing_records), width="stretch")

    with catalog_tab3:
        st.markdown("##### 🧮 Calculate Estimated Cost with Custom Rates")
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_model = st.selectbox("Select Baseline Model", list(DEFAULT_MODEL_PRICING.keys()), index=0, key="sim_m_sel")
            sim_in_toks = st.number_input("Input Tokens", min_value=0, value=1500, step=100, key="sim_in_t")
        with sim_c2:
            cur_pricing = DEFAULT_MODEL_PRICING[sim_model]
            sim_in_price = st.number_input("Input Price ($ / 1M tokens)", value=cur_pricing.input_price_per_mtoken, step=0.1, key="sim_in_p")
            sim_out_toks = st.number_input("Output Tokens", min_value=0, value=350, step=50, key="sim_out_t")
        with sim_c3:
            sim_out_price = st.number_input("Output Price ($ / 1M tokens)", value=cur_pricing.output_price_per_mtoken, step=0.1, key="sim_out_p")
            sim_tool_calls = st.number_input("Tool Invocations (Search API)", min_value=0, value=1, step=1, key="sim_tc_cnt")

        sim_llm_cost = (sim_in_toks / 1_000_000 * sim_in_price) + (sim_out_toks / 1_000_000 * sim_out_price)
        sim_tool_cost = sim_tool_calls * DEFAULT_TOOL_PRICING.get("web_search", 0.005)
        sim_tot_cost = sim_llm_cost + sim_tool_cost

        st.info(f"💵 **Calculated Estimated Cost:** `${sim_tot_cost:.6f}` (LLM: `${sim_llm_cost:.6f}` + Tool: `${sim_tool_cost:.6f}`)")
