"""
Section 10: Tool Analytics - Tool Sequence Flow, Efficiency, 7-Dimension Evaluation Suite, and Telemetry.
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.core.entities import Trace
from src.evaluation.tool_evaluators import (
    extract_tool_telemetry,
    ALL_TOOL_EVALUATORS,
)
from src.evaluation.mcp_evaluators import (
    extract_mcp_telemetry,
    ALL_MCP_EVALUATORS,
)
from dashboard.views.common import steps_to_spans, render_kpi_card, render_section_header


def render_tool_analytics(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    load_steps_fn: Any,
    tc_manager: Any,
):
    render_section_header(
        title="Tool & MCP Execution Telemetry",
        subtitle="Inspect tool selection accuracy, Model Context Protocol (MCP) server calls, argument correctness, execution errors, and unnecessary call detection.",
        breadcrumb="OBSERVABILITY // TOOL & MCP CALLS",
        action_badge="CALL TELEMETRY",
    )

    if filtered_runs.empty:
        st.info("No runs match active filters.")
        return

    # Top KPI Metrics across filtered runs
    tool_eval_names = [
        "tool_accuracy", "tool_selection_accuracy", "tool_argument_correctness",
        "tool_execution_success", "tool_efficiency", "unnecessary_tool_calls",
        "tool_sequence_correctness", "tool_retry_behavior",
        "mcp_tool_selection", "mcp_argument_correctness", "mcp_tool_success",
        "unnecessary_mcp_calls", "mcp_latency", "mcp_failure_analysis",
        "mcp_tool_sequence", "mcp_evaluation_suite",
    ]
    tool_evals_df = filtered_evals[filtered_evals["metric_name"].isin(tool_eval_names)] if not filtered_evals.empty else pd.DataFrame()

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        render_kpi_card("Tool & MCP Assertions", str(len(tool_evals_df)), subtitle="Evaluated Checks", icon="🛠️", accent_color="#38bdf8")
    with t2:
        acc_df = tool_evals_df[tool_evals_df["metric_name"].str.contains("selection|accuracy", case=False, na=False)]
        avg_acc = (acc_df["passed"].mean() * 100) if not acc_df.empty else 0.0
        render_kpi_card("Selection Accuracy", f"{avg_acc:.1f}%", health="🟢 Healthy" if avg_acc >= 90 else ("🟡 Degraded" if avg_acc >= 75 else "🔴 Critical"), icon="🎯", accent_color="#10b981")
    with t3:
        exec_df = tool_evals_df[tool_evals_df["metric_name"].str.contains("execution|success", case=False, na=False)]
        avg_exec = (exec_df["passed"].mean() * 100) if not exec_df.empty else 100.0
        render_kpi_card("Execution Success", f"{avg_exec:.1f}%", health="🟢 Healthy" if avg_exec >= 90 else "🔴 Critical", icon="⚡", accent_color="#818cf8")
    with t4:
        eff_df = tool_evals_df[tool_evals_df["metric_name"].str.contains("unnecessary|efficiency", case=False, na=False)]
        avg_eff = (eff_df["passed"].mean() * 100) if not eff_df.empty else 100.0
        render_kpi_card("Protocol Efficiency", f"{avg_eff:.1f}%", subtitle="Zero Unnecessary Calls", health="🟢 Healthy" if avg_eff >= 90 else "🟡 Degraded", icon="✨", accent_color="#a855f7")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Select Run for In-Depth Tool Telemetry
    st.subheader("🔍 Inspect Single Run Tool & MCP Telemetry")
    all_run_ids = filtered_runs["run_id"].tolist()
    
    # Pick a preferred tool run by default if not set by drill-down
    default_rid = st.session_state.get("selected_run_id")
    if not default_rid or default_rid not in all_run_ids:
        tool_candidate_runs = filtered_runs[filtered_runs["tools_called"].astype(str).str.len() > 4]
        if not tool_candidate_runs.empty:
            default_rid = tool_candidate_runs.iloc[0]["run_id"]
        else:
            default_rid = all_run_ids[0]

    selected_run_id = st.selectbox(
        "Select Run ID for Tool Analysis:",
        all_run_ids,
        index=all_run_ids.index(default_rid),
        key="tool_run_selector"
    )

    run_row = filtered_runs[filtered_runs["run_id"] == selected_run_id].iloc[0]
    steps = load_steps_fn(int(selected_run_id))
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

    tool_summary = extract_tool_telemetry(run_trace, run_tc)
    mcp_calls = extract_mcp_telemetry(run_trace, run_tc)
    has_tools = tool_summary.total_tool_calls > 0 or (run_tc and bool(run_tc.expected_tools or run_tc.expected_tool_sequence or run_tc.expected_mcp_tools or run_tc.expected_mcp_server))

    if not has_tools:
        st.info(f"Run #{selected_run_id} did not execute any tools and has no expected tools configured.")
        return

    # 1. MCP Specific Flow (if MCP calls present or expected)
    if mcp_calls or (run_tc and (run_tc.expected_mcp_server or run_tc.expected_mcp_tools)):
        st.markdown("#### 🔌 Model Context Protocol (MCP) Trace Flow")
        exp_mcp_server = run_tc.expected_mcp_server or "Any"
        exp_mcp_tools = " ➔ ".join([f"`{t}`" for t in (run_tc.expected_mcp_tools or run_tc.expected_tools or [])]) or "`none`"
        
        act_mcp_servers = list(dict.fromkeys([c.mcp_server for c in mcp_calls if c.mcp_server]))
        act_mcp_srv_str = ", ".join(act_mcp_servers) if act_mcp_servers else "None"
        act_mcp_tools = " ➔ ".join([f"`{c.tool_name}`" for c in mcp_calls]) if mcp_calls else "`none`"

        with st.container(border=True):
            st.markdown(
                f""" <div style="font-size: 11px; font-weight: 700; color: #a855f7; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;"> 🔌 MCP Bridge & Tool Hierarchy: Agent ➔ MCP Server ➔ Tool ➔ Result ➔ LLM </div> <div style="margin-bottom: 8px; font-size: 13px;"> <strong style="color: #60a5fa;">📋 Expected MCP Flow:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: #cbd5e1;">Agent ➔ Server: [{exp_mcp_server}] ➔ Tool: {exp_mcp_tools} ➔ Result ➔ LLM</span> </div> <div style="font-size: 13px;"> <strong style="color: #a855f7;">⚡ Actual MCP Flow:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0;">Agent ➔ Server: [{act_mcp_srv_str}] ➔ Tool: {act_mcp_tools} ➔ Result ➔ LLM</span> </div> """,
                unsafe_allow_html=True,
            )

        # MCP KPI Highlights
        mcp_kpi1, mcp_kpi2, mcp_kpi3, mcp_kpi4 = st.columns(4)
        with mcp_kpi1:
            render_kpi_card("MCP Invocations", str(len(mcp_calls)), subtitle="Calls Recorded", icon="🔌", accent_color="#a855f7")
        with mcp_kpi2:
            mcp_succ = sum(1 for c in mcp_calls if c.execution_status == "success")
            mcp_succ_pct = (mcp_succ / len(mcp_calls) * 100) if mcp_calls else 100.0
            render_kpi_card("MCP Success Rate", f"{mcp_succ_pct:.0f}%", health="🟢 Healthy" if mcp_succ_pct >= 90 else "🔴 Critical", icon="⚡", accent_color="#10b981")
        with mcp_kpi3:
            mcp_lat = sum(c.latency for c in mcp_calls)
            render_kpi_card("MCP Latency", f"{mcp_lat:.1f} ms", subtitle="Roundtrip Overhead", icon="⏱️", accent_color="#c084fc")
        with mcp_kpi4:
            render_kpi_card("MCP Servers", str(len(act_mcp_servers)), subtitle=f"{act_mcp_srv_str[:20]}", icon="🌐", accent_color="#38bdf8")

    # 2. General Tool Sequence Flow Comparison (Expected vs Actual)
    exp_flow_str = " ➔ ".join([f"`{t}`" for t in tool_summary.expected_tool_sequence]) if tool_summary.expected_tool_sequence else "`none`"
    act_flow_str = " ➔ ".join([f"`{t}`" for t in tool_summary.actual_tool_sequence]) if tool_summary.actual_tool_sequence else "`none`"

    with st.container(border=True):
        st.markdown(
            f""" <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;"> 🔄 Overall Tool Execution Sequence </div> <div style="margin-bottom: 8px; font-size: 13px;"> <strong style="color: #60a5fa;">📋 Expected:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: #cbd5e1;">LLM ➔ {exp_flow_str} ➔ Final Answer</span> </div> <div style="font-size: 13px;"> <strong style="color: {'#34d399' if tool_summary.is_selection_accurate and tool_summary.unnecessary_tool_calls_count == 0 else '#f87171'};">⚡ Actual:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: #cbd5e1;">LLM ➔ {act_flow_str} ➔ Final Answer</span> </div> """,
            unsafe_allow_html=True,
        )

    # 3. Unnecessary Tool Calls Alert Banner
    if tool_summary.unnecessary_tool_calls_count > 0:
        st.warning(f"⚠️ **{tool_summary.unnecessary_tool_calls_count} potentially unnecessary tool call{'s' if tool_summary.unnecessary_tool_calls_count != 1 else ''} detected**")
        with st.expander("🔍 Unnecessary Tool Calls Breakdown", expanded=True):
            for flag in tool_summary.unnecessary_flags:
                st.markdown(f"- ⚠️ {flag}")
    else:
        st.success("✅ **Tool Efficiency:** Zero unnecessary tool calls detected.")

    # 4. Tool Selection Accuracy Diagnosis Card
    exp_tool_display = ", ".join(run_tc.expected_tools) if (run_tc and run_tc.expected_tools) else "none"
    act_tool_display = ", ".join(tool_summary.tools_called_names) if tool_summary.tools_called_names else "none"

    if not tool_summary.is_selection_accurate:
        st.error(
            f"**Tool Selection Result:** `FAILED` &nbsp;|&nbsp; "
            f"**Expected Tool(s):** `{exp_tool_display}` &nbsp;|&nbsp; "
            f"**Actual Tool(s):** `{act_tool_display}`\n\n"
            f"**Reason:** {tool_summary.selection_reason}"
        )
    else:
        st.success(
            f"**Tool Selection Result:** `PASSED` &nbsp;|&nbsp; "
            f"**Expected:** `{exp_tool_display}` &nbsp;|&nbsp; "
            f"**Actual:** `{act_tool_display}`"
        )

    # 5. Summary Metric Cards
    tc_col1, tc_col2, tc_col3, tc_col4 = st.columns(4)
    with tc_col1:
        render_kpi_card("Tool Invocations", str(tool_summary.total_tool_calls), subtitle="Spans in Trace", icon="🛠️", accent_color="#38bdf8")
    with tc_col2:
        success_rate = (tool_summary.successful_tool_calls / tool_summary.total_tool_calls * 100) if tool_summary.total_tool_calls else 100.0
        s_health = "🟢 Healthy" if success_rate >= 90 else "🔴 Critical"
        render_kpi_card("Execution Success", f"{success_rate:.0f}%", subtitle="Completed Spans", health=s_health, icon="⚡", accent_color="#10b981")
    with tc_col3:
        render_kpi_card("Tool Latency", f"{tool_summary.total_tool_latency_ms:.0f} ms", subtitle="Cumulative Runtime", icon="⏱️", accent_color="#818cf8")
    with tc_col4:
        r_health = "🟢 Healthy" if tool_summary.retry_loops_detected == 0 else "🔴 Critical"
        render_kpi_card("Retry Loops", str(tool_summary.retry_loops_detected), subtitle="Loop Anomalies", health=r_health, icon="🔁", accent_color="#06b6d4")

    st.markdown("<div style='margin-top: 14px; margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # 6. Captured Tool Call Telemetry Table (All 10 dimensions + MCP)
    if tool_summary.tool_calls:
        st.markdown("##### 🔍 Captured Tool & MCP Call Telemetry (All Dimensions):")
        call_rows = []
        for call in tool_summary.tool_calls:
            arg_disp = json.dumps(call.arguments) if isinstance(call.arguments, (dict, list)) else str(call.arguments)
            if len(arg_disp) > 40:
                arg_disp = arg_disp[:37] + "..."

            exp_arg_disp = json.dumps(call.expected_arguments) if isinstance(call.expected_arguments, (dict, list)) else str(call.expected_arguments or "—")
            if len(exp_arg_disp) > 30:
                exp_arg_disp = exp_arg_disp[:27] + "..."

            resp_disp = str(call.response or "—")
            if len(resp_disp) > 45:
                resp_disp = resp_disp[:42] + "..."

            status_icon = "🟢 Success" if call.execution_status == "success" else f"🔴 {call.execution_status.capitalize()}"
            unnec_icon = "⚠️ Yes" if call.is_unnecessary else "✅ No"
            mcp_tag = f"🔌 {call.mcp_server or 'MCP'}" if call.is_mcp else "🛠️ Internal"

            call_rows.append({
                "Pos": f"#{call.sequence_position}",
                "Type": mcp_tag,
                "Tool Name": call.tool_name,
                "Expected Tool": call.expected_tool or "—",
                "Arguments": arg_disp,
                "Expected Args": exp_arg_disp,
                "Status": status_icon,
                "Response Preview": resp_disp,
                "Latency (ms)": f"{call.latency:.1f}",
                "Retries": call.retry_count,
                "Error": call.error or "None",
                "Unnecessary?": unnec_icon,
            })
        st.dataframe(pd.DataFrame(call_rows), width="stretch")

    # 7. Dedicated 7-Dimension MCP Evaluation Suite Table (if MCP calls present or expected)
    if mcp_calls or (run_tc and (run_tc.expected_mcp_server or run_tc.expected_mcp_tools)):
        st.markdown("##### 🔌 7-Dimension Model Context Protocol (MCP) Evaluation Suite:")
        mcp_eval_records = []
        for eval_cls in ALL_MCP_EVALUATORS:
            eval_instance = eval_cls()
            ev_res = eval_instance.evaluate(test_case=run_tc, execution_result=run_row.get("final_answer"), trace=run_trace)
            mcp_eval_records.append({
                "MCP Metric": ev_res.metric_name,
                "Score": f"{ev_res.score:.2f}",
                "Verdict": "✅ PASS" if ev_res.passed else "❌ FAIL",
                "Threshold": f"{ev_res.threshold:.2f}",
                "Type": ev_res.evaluator_type,
                "Explanation": ev_res.explanation,
            })
        st.dataframe(pd.DataFrame(mcp_eval_records), width="stretch")

    # 8. Detailed 7-Dimension Tool Evaluators Table
    st.markdown("##### 📊 Standard 7-Dimension Tool Evaluation Suite:")
    tool_eval_records = []
    for eval_cls in ALL_TOOL_EVALUATORS:
        eval_instance = eval_cls()
        ev_res = eval_instance.evaluate(test_case=run_tc, execution_result=run_row.get("final_answer"), trace=run_trace)
        tool_eval_records.append({
            "Metric": ev_res.metric_name,
            "Score": f"{ev_res.score:.2f}",
            "Verdict": "✅ PASS" if ev_res.passed else "❌ FAIL",
            "Threshold": f"{ev_res.threshold:.2f}",
            "Type": ev_res.evaluator_type,
            "Explanation": ev_res.explanation,
        })
    st.dataframe(pd.DataFrame(tool_eval_records), width="stretch")
