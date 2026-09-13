"""
Section 12: Professional Evaluation Report Generator & Benchmark Suite.

Generates comprehensive 17-section Evaluation Reports with executive KPI scorecards,
quality gate validation, regression analysis, root cause categorization, and
downloadable multi-format exports (HTML/PDF, JSON, CSV).
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.storage.db import get_session
from src.storage.models import Experiment as ExperimentRow, Run
from src.core.dataset import DatasetLoader, DEFAULT_GOLDEN_PATH
from src.core.engine import EvaluationEngine
from src.registry.registry import AgentRegistry
from src.evaluation.scoring import ScoringConfig
from src.analysis.report_generator import EvaluationReportGenerator, EvaluationReportData
from dashboard.views.common import render_kpi_card, render_section_header


def render_reports(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    all_runs: pd.DataFrame,
    all_evals: pd.DataFrame,
    registry: AgentRegistry,
    scoring_config: ScoringConfig,
    all_tags: List[str],
):
    render_section_header(
        title="Evaluation Reports & Audit Generator",
        subtitle="Generate executive summary scorecards, verify quality gates, and export comprehensive 17-section evaluation audits in HTML/PDF, JSON, or CSV.",
        breadcrumb="EXECUTIVE // AUDIT REPORTS",
        action_badge="AUDIT READY",
    )

    rep_tab1, rep_tab2 = st.tabs([
        "📑 Professional Evaluation Report",
        "🚀 Automated Benchmark Runner",
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: Professional Evaluation Report Generator
    # ---------------------------------------------------------------------------
    with rep_tab1:
        # 1. Experiment & Scope Selector
        session = get_session()
        exp_list = []
        try:
            db_exps = session.query(ExperimentRow).order_by(ExperimentRow.created_at.desc()).all()
            exp_list = [(e.id, f"{e.name} ({e.id[:8]}... • {e.created_at.strftime('%Y-%m-%d %H:%M') if e.created_at else 'Recent'})") for e in db_exps]
        except Exception:
            pass
        finally:
            session.close()

        scope_col1, scope_col2, scope_col3 = st.columns([2, 2, 1])
        with scope_col1:
            exp_options = ["Active Global Filter Scope"] + [e[1] for e in exp_list]
            selected_exp_label = st.selectbox(
                "Select Evaluation Experiment / Suite*",
                exp_options,
                index=0,
                key="rep_sel_exp_label",
                help="Choose a recorded batch experiment or use the active filter selection from the sidebar."
            )

        with scope_col2:
            base_options = ["None (Auto-detect previous suite)"] + [e[1] for e in exp_list]
            selected_base_label = st.selectbox(
                "Baseline Experiment for Regression Diff (Optional)",
                base_options,
                index=0,
                key="rep_sel_base_label",
                help="Select an earlier experiment to calculate score deltas, newly failing tests, and regression status."
            )

        with scope_col3:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            refresh_btn = st.button("🔄 Refresh Report", width="stretch", key="btn_refresh_rep")

        # Resolve selected experiment_id and baseline_id
        target_exp_id = None
        target_run_ids = None
        if selected_exp_label != "Active Global Filter Scope":
            for eid, elbl in exp_list:
                if elbl == selected_exp_label:
                    target_exp_id = eid
                    break
        else:
            if not filtered_runs.empty:
                target_run_ids = filtered_runs["run_id"].tolist()

        target_base_id = None
        if selected_base_label != "None (Auto-detect previous suite)":
            for eid, elbl in exp_list:
                if elbl == selected_base_label:
                    target_base_id = eid
                    break

        # Generate Report Data using EvaluationReportGenerator
        with st.spinner("Generating 17-section Evaluation Audit Report..."):
            try:
                report_data: EvaluationReportData = EvaluationReportGenerator.generate_report_data(
                    experiment_id=target_exp_id,
                    run_ids=target_run_ids,
                    baseline_experiment_id=target_base_id,
                    scoring_config=scoring_config,
                )
            except Exception as e:
                st.error(f"Failed to generate evaluation report: {str(e)}")
                return

        s = report_data.summary
        gate_passed = s.quality_gate_status == "PASSED"
        gate_color = "#10b981" if gate_passed else "#ef4444"
        gate_bg = "rgba(16, 185, 129, 0.12)" if gate_passed else "rgba(239, 68, 68, 0.12)"

        # ---------------------------------------------------------------------------
        # EXECUTIVE SUMMARY SCORECARD (Matching Specification)
        # ---------------------------------------------------------------------------
        st.markdown(
            f""" <div style=" background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85)); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 24px; margin-top: 10px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4); "> <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 14px; margin-bottom: 18px;"> <div> <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8;">EXECUTIVE EVALUATION SUMMARY</div> <div style="font-size: 22px; font-weight: 800; color: #ffffff; margin-top: 2px;"> {s.agent_name} <span style="font-weight: 400; color: #94a3b8; font-size: 16px;">({s.version})</span> </div> </div> <div style="display: flex; align-items: center; gap: 12px;"> <span style=" padding: 6px 16px; border-radius: 6px; font-weight: 800; font-size: 14px; background: {gate_bg}; color: {gate_color}; border: 1px solid {gate_color}44; letter-spacing: 0.5px; "> QUALITY GATE: {s.quality_gate_status} </span> </div> </div> <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px;"> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Overall Score</div> <div style="font-size: 20px; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.overall_score_str}</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Task Success</div> <div style="font-size: 20px; font-weight: 800; color: #10b981; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.task_success_pct:.1f}%</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Tool Accuracy</div> <div style="font-size: 20px; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.tool_accuracy_pct:.1f}%</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Groundedness</div> <div style="font-size: 20px; font-weight: 800; color: #a855f7; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.groundedness_pct:.1f}%</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">P95 Latency</div> <div style="font-size: 20px; font-weight: 800; color: #f59e0b; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.p95_latency_sec:.2f}s</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Est. Cost / Run</div> <div style="font-size: 20px; font-weight: 800; color: #34d399; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">${s.estimated_cost_per_run:.4f}</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Regressions</div> <div style="font-size: 20px; font-weight: 800; color: {'#ef4444' if s.regressions_count > 0 else '#10b981'}; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.regressions_count}</div> </div> <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 12px;"> <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Critical Failures</div> <div style="font-size: 20px; font-weight: 800; color: {'#ef4444' if s.critical_failures_count > 0 else '#10b981'}; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{s.critical_failures_count}</div> </div> </div> </div> """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------------------------------------
        # MULTI-FORMAT DOWNLOAD TOOLBAR
        # ---------------------------------------------------------------------------
        st.subheader("📥 Export & Download Multi-Format Reports")
        st.caption("Download the complete evaluation audit in print-ready HTML (saveable as PDF in browser), structured JSON, or tabular CSV format.")

        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        
        # 1. HTML / PDF Export
        html_report = EvaluationReportGenerator.export_html(report_data)
        with d_c1:
            st.download_button(
                label="📄 Download HTML / PDF Report",
                data=html_report.encode("utf-8"),
                file_name=f"evaluation_report_{report_data.report_id}.html",
                mime="text/html",
                width="stretch",
                type="primary",
            )

        # 2. JSON Export
        json_report = EvaluationReportGenerator.export_json(report_data)
        with d_c2:
            st.download_button(
                label="📦 Download JSON Report",
                data=json_report.encode("utf-8"),
                file_name=f"evaluation_report_{report_data.report_id}.json",
                mime="application/json",
                width="stretch",
            )

        # 3. CSV Exports
        csv_files = EvaluationReportGenerator.export_csv(report_data)
        with d_c3:
            st.download_button(
                label="📊 Download Test Runs (CSV)",
                data=csv_files["test_runs.csv"].encode("utf-8"),
                file_name=f"evaluation_test_runs_{report_data.report_id}.csv",
                mime="text/csv",
                width="stretch",
            )
        with d_c4:
            st.download_button(
                label="📈 Download Metrics (CSV)",
                data=csv_files["metrics.csv"].encode("utf-8"),
                file_name=f"evaluation_metrics_{report_data.report_id}.csv",
                mime="text/csv",
                width="stretch",
            )

        st.divider()

        # ---------------------------------------------------------------------------
        # INTERACTIVE 17-SECTION DIAGNOSTIC EXPLORER
        # ---------------------------------------------------------------------------
        st.subheader("🔍 Interactive 17-Section Diagnostic Explorer")
        st.caption("Detailed audit breakdown across all 17 standardized dimensions.")

        diag_tab1, diag_tab2, diag_tab3, diag_tab4, diag_tab5, diag_tab6 = st.tabs([
            "🏛️ Overview & Metadata (Sec 1-8)",
            "🧪 Test Executions (Sec 9-10)",
            "⚖️ Regression Analysis (Sec 11)",
            "⏱️ Latency & Cost (Sec 12-13)",
            "🛠️ Tool & RAG Analytics (Sec 14-15)",
            "🚨 RCA & Action Items (Sec 16-17)",
        ])

        # Sub-tab 1: Sections 1-8
        with diag_tab1:
            st.markdown("#### 1-5. Agent, Model, Dataset & Timestamp Metadata")
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                meta_df = pd.DataFrame([
                    {"Parameter": "1. Agent Name & ID", "Value": f"{report_data.sec1_agent_info.name} ({report_data.sec1_agent_info.agent_id})"},
                    {"Parameter": "2. Agent Version", "Value": report_data.sec2_agent_version.version},
                    {"Parameter": "3. Primary Model", "Value": f"{report_data.sec3_model_info.primary_model} ({report_data.sec3_model_info.provider})"},
                    {"Parameter": "4. Dataset Suite", "Value": f"{report_data.sec4_dataset_info.name} (v{report_data.sec4_dataset_info.version})"},
                    {"Parameter": "5. Execution Duration", "Value": f"{report_data.sec5_timestamp_info.duration_formatted} ({report_data.sec5_timestamp_info.start_time[:19]} UTC)"},
                ])
                st.dataframe(meta_df, width="stretch", hide_index=True)

            with m_c2:
                st.markdown("##### 6-7. Quality Gate Decision Criteria")
                q_rules_df = pd.DataFrame(report_data.sec7_quality_gate.rules_evaluated)
                if not q_rules_df.empty:
                    q_rules_df["status"] = q_rules_df["passed"].apply(lambda p: "🟢 PASSED" if p else "🔴 FAILED")
                    st.dataframe(q_rules_df[["name", "target", "actual", "status"]], width="stretch", hide_index=True)

            st.markdown("#### 8. Metric Breakdown & Assertion Pass Rates")
            metrics_df = pd.DataFrame([asdict(m) for m in report_data.sec8_metric_breakdown])
            if not metrics_df.empty:
                st.dataframe(
                    metrics_df[[
                        "display_name", "evaluator_type", "pass_rate_pct", "avg_score", "threshold", "passed_checks", "total_checks", "health"
                    ]],
                    column_config={
                        "display_name": "Metric",
                        "evaluator_type": "Evaluator",
                        "pass_rate_pct": st.column_config.ProgressColumn("Pass Rate (%)", format="%.1f%%", min_value=0, max_value=100),
                        "avg_score": st.column_config.NumberColumn("Avg Score", format="%.3f"),
                        "threshold": "Threshold",
                        "passed_checks": "Passed",
                        "total_checks": "Total",
                        "health": "Health",
                    },
                    width="stretch",
                    hide_index=True,
                )

        # Sub-tab 2: Sections 9 & 10
        with diag_tab2:
            st.markdown(f"#### 10. Failed Test Cases ({len(report_data.sec10_failed_tests)})")
            if report_data.sec10_failed_tests:
                failed_df = pd.DataFrame([
                    {
                        "Run ID": ft.run_id,
                        "Task ID": ft.task_id,
                        "Query": ft.query,
                        "Failure Category": ft.failure_category or "Assertion Failure",
                        "Latency (s)": ft.latency_sec,
                        "Cost ($)": ft.cost_usd,
                        "Failed Metrics": ", ".join(ft.failed_metrics),
                        "Error Logs": ft.error_message or "Metric threshold breached",
                    }
                    for ft in report_data.sec10_failed_tests
                ])
                st.dataframe(failed_df, width="stretch", hide_index=True)

                st.markdown("##### 🔬 Failure Analysis for Failed Test Cases")
                for ft in report_data.sec10_failed_tests:
                    with st.expander(f"🔬 Run #{ft.run_id} — {ft.task_id}: {ft.failure_category or 'Failed'}", expanded=True):
                        if getattr(ft, "failure_analysis", None):
                            fa = ft.failure_analysis
                            st.markdown(
                                f""" <div style="background:rgba(15,23,42,0.85);border:1px solid #f43f5e44;border-radius:10px;padding:14px 16px;margin-bottom:8px;"> <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:2px;">FAILURE</div> <div style="font-size:15px;font-weight:800;color:#f8fafc;margin-bottom:10px;">{fa.get('failure_type', ft.failure_category or 'Failure')}</div> <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;"> <div style="background:rgba(56,189,248,0.06);padding:8px 12px;border-radius:6px;border:1px solid rgba(56,189,248,0.18);"> <div style="font-size:10px;font-weight:700;color:#38bdf8;">📋 EXPECTED</div> <div style="font-size:12px;font-family:'JetBrains Mono',monospace;color:#f8fafc;">{fa.get('expected', 'N/A')}</div> </div> <div style="background:rgba(244,63,94,0.06);padding:8px 12px;border-radius:6px;border:1px solid rgba(244,63,94,0.18);"> <div style="font-size:10px;font-weight:700;color:#fb7185;">💥 ACTUAL</div> <div style="font-size:12px;font-family:'JetBrains Mono',monospace;color:#f8fafc;">{fa.get('actual', 'N/A')}</div> </div> </div> <div style="background:rgba(255,255,255,0.03);padding:8px 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.07);margin-bottom:8px;"> <div style="font-size:10px;font-weight:700;color:#94a3b8;">🔍 EVIDENCE <span style="font-weight:500;color:#64748b;">(Observed Telemetry)</span></div> <div style="font-size:12px;color:#cbd5e1;font-family:'JetBrains Mono',monospace;">{fa.get('evidence', 'N/A')}</div> </div> <div style="background:rgba(168,85,247,0.06);padding:8px 12px;border-radius:6px;border:1px solid rgba(168,85,247,0.16);margin-bottom:8px;"> <div style="font-size:10px;font-weight:700;color:#c084fc;">🧠 LIKELY ROOT CAUSE <span style="font-weight:500;color:#8b5cf6;">(Inferred Hypothesis)</span></div> <div style="font-size:12px;color:#e4e4e7;">{fa.get('root_cause', 'N/A')}</div> </div> <div style="background:rgba(16,185,129,0.06);padding:8px 12px;border-radius:6px;border:1px solid rgba(16,185,129,0.16);"> <div style="font-size:10px;font-weight:700;color:#34d399;">💡 RECOMMENDATION</div> <div style="font-size:12px;color:#e4e4e7;">{fa.get('recommendation', 'N/A')}</div> </div> </div> """,
                                unsafe_allow_html=True,
                            )
                            if fa.get("formatted_text"):
                                with st.expander("📋 Copy Plaintext", expanded=False):
                                    st.code(fa["formatted_text"], language="yaml")
                        else:
                            st.info(f"Failed metrics: {', '.join(ft.failed_metrics)}")
            else:
                st.success("🎉 Zero failed tests! All evaluated test cases passed successfully.")

            st.markdown(f"#### 9. Passed Test Cases ({len(report_data.sec9_passed_tests)})")
            if report_data.sec9_passed_tests:
                passed_df = pd.DataFrame([
                    {
                        "Run ID": pt.run_id,
                        "Task ID": pt.task_id,
                        "Query": pt.query,
                        "Score (%)": pt.overall_score,
                        "Latency (s)": pt.latency_sec,
                        "Cost ($)": pt.cost_usd,
                        "Tokens": pt.tokens,
                        "Status": "🟢 PASSED",
                    }
                    for pt in report_data.sec9_passed_tests
                ])
                st.dataframe(passed_df, width="stretch", hide_index=True)

        # Sub-tab 3: Section 11
        with diag_tab3:
            st.markdown("#### 11. Regression Analysis & Transition Matrix")
            reg = report_data.sec11_regression_analysis
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric("Baseline Suite", reg.baseline_name)
            r_c2.metric("Newly Failing Tests", str(reg.newly_failing_count), delta=f"-{reg.newly_failing_count}" if reg.newly_failing_count > 0 else None, delta_color="inverse")
            r_c3.metric("Newly Passing Tests", str(reg.newly_passing_count), delta=f"+{reg.newly_passing_count}" if reg.newly_passing_count > 0 else None)
            r_c4.metric("Persistent Failures", str(reg.persistent_failures_count))

            if reg.metric_deltas:
                st.markdown("##### Metric Deltas vs Baseline")
                st.dataframe(pd.DataFrame(reg.metric_deltas), width="stretch", hide_index=True)

        # Sub-tab 4: Sections 12 & 13
        with diag_tab4:
            st.markdown("#### 12-13. Latency & Cost Distribution")
            lat = report_data.sec12_latency_analysis
            cost = report_data.sec13_cost_analysis

            l_c1, l_c2, l_c3, l_c4 = st.columns(4)
            l_c1.metric("Mean Latency", f"{lat.mean_latency_ms:.0f} ms", f"{lat.mean_latency_sec:.2f}s")
            l_c2.metric("P95 Latency", f"{lat.p95_latency_ms:.0f} ms", f"{lat.p95_latency_sec:.2f}s")
            l_c3.metric("Budget Compliance", f"{lat.budget_compliance_pct:.1f}%")
            l_c4.metric("Total Cost", f"${cost.total_cost_usd:.4f}", f"${cost.cost_per_run_usd:.4f} / run")

            st.markdown("##### Slowest Test Cases")
            if lat.slowest_tests:
                st.dataframe(pd.DataFrame(lat.slowest_tests), width="stretch", hide_index=True)

        # Sub-tab 5: Sections 14 & 15
        with diag_tab5:
            st.markdown("#### 14-15. Tool Execution & RAG Grounding Analytics")
            tool = report_data.sec14_tool_analysis
            rag = report_data.sec15_rag_analysis

            t_c1, t_c2, t_c3, t_c4 = st.columns(4)
            t_c1.metric("Tool Accuracy", f"{tool.tool_selection_accuracy_pct:.1f}%")
            t_c2.metric("Execution Success", f"{tool.tool_execution_success_pct:.1f}%")
            t_c3.metric("RAG Faithfulness", f"{rag.answer_faithfulness_pct:.1f}%")
            t_c4.metric("Context Precision", f"{rag.context_precision_pct:.1f}%")

            if tool.tool_usage_breakdown:
                st.markdown("##### Tool Invocations Breakdown")
                st.bar_chart(pd.DataFrame(list(tool.tool_usage_breakdown.items()), columns=["Tool", "Invocations"]).set_index("Tool"))

        # Sub-tab 6: Sections 16 & 17
        with diag_tab6:
            st.markdown(f"#### 16. Root-Cause Categorization ({report_data.sec16_root_cause_analysis.critical_failures_count} Critical Failures)")
            rca = report_data.sec16_root_cause_analysis
            if rca.failure_category_counts:
                rca_df = pd.DataFrame(list(rca.failure_category_counts.items()), columns=["Failure Category", "Occurrences"])
                st.dataframe(rca_df, width="stretch", hide_index=True)
            else:
                st.info("No failure patterns identified in the selected evaluation run set.")

            st.markdown(f"#### 17. Actionable Engineering Recommendations ({len(report_data.sec17_recommendations)})")
            for rec in report_data.sec17_recommendations:
                p_color = "#ef4444" if "P0" in rec.priority else ("#f59e0b" if "P1" in rec.priority else "#38bdf8")
                with st.expander(f"**[{rec.priority}] {rec.title}** ({rec.category})", expanded=True):
                    st.markdown(f"**Rationale:** {rec.rationale}")
                    st.markdown("**Action Items:**")
                    for a in rec.action_items:
                        st.markdown(f"- {a}")

        # Embedded HTML Report Preview
        with st.expander("👁️ View Live Standalone HTML Report Preview", expanded=False):
            st.components.v1.html(html_report, height=800, scrolling=True)

    # ---------------------------------------------------------------------------
    # TAB 2: Automated Benchmark Runner
    # ---------------------------------------------------------------------------
    with rep_tab2:
        st.subheader("🚀 Trigger Automated Benchmark Suite")
        st.caption("Execute batch evaluation of any registered agent across the golden dataset or custom test cases.")

        active_agents = registry.list_agents(status="active")
        if not active_agents:
            st.warning("No active agents found in the registry. Please register or enable an agent in the **🤖 Agents** section first.")
        else:
            run_form_c1, run_form_c2 = st.columns(2)
            with run_form_c1:
                eval_agent_name = st.selectbox("Select Agent to Evaluate*", [a.name for a in active_agents], key="rep_agent_name")
                target_agent = registry.get_agent(eval_agent_name)
                agent_versions = [v.version for v in registry.get_agent_versions(target_agent.agent_id)] if target_agent else []
                eval_version = st.selectbox("Select Version", agent_versions, index=agent_versions.index(target_agent.active_version) if target_agent and target_agent.active_version in agent_versions else 0, key="rep_ver_name")

            with run_form_c2:
                dataset_choice = st.selectbox(
                    "Select Evaluation Dataset*",
                    [
                        "golden_tasks (Baseline 15 Tasks)",
                        "Active Database Test Cases (All Enabled)",
                        "Filter by Tag from Database",
                        "Custom JSON File Path",
                    ],
                    key="rep_dataset_choice"
                )
                tag_to_run = None
                if dataset_choice == "Custom JSON File Path":
                    dataset_path = st.text_input("Dataset JSON File Path*", value=DEFAULT_GOLDEN_PATH, key="rep_custom_ds_path")
                elif dataset_choice == "Filter by Tag from Database":
                    tag_to_run = st.selectbox("Select Tag to Run", all_tags, key="rep_tag_run")
                else:
                    dataset_path = DEFAULT_GOLDEN_PATH

                custom_exp_name = st.text_input("Experiment Name (optional)", placeholder="e.g. Sprint_14_Prompt_Tuning", key="rep_exp_name")

            run_col_opt1, run_col_opt2 = st.columns(2)
            with run_col_opt1:
                eval_live_mode = st.checkbox("Live Agent Mode (Anthropic Claude Calls)", value=False, help="Requires ANTHROPIC_API_KEY in .env", key="rep_live_mode")
            with run_col_opt2:
                judge_choice = st.selectbox(
                    "⚖️ LLM Judge Mode",
                    [
                        "Auto (Live with Mock Fallback)",
                        "OpenAI (GPT-4o-mini)",
                        "Anthropic (Claude-3.5-Haiku)",
                        "Deterministic Mock",
                    ],
                    index=0,
                    key="rep_judge_choice"
                )
                judge_prov_map = {
                    "Auto (Live with Mock Fallback)": "auto",
                    "OpenAI (GPT-4o-mini)": "openai",
                    "Anthropic (Claude-3.5-Haiku)": "anthropic",
                    "Deterministic Mock": "mock",
                }
                selected_judge_prov = judge_prov_map.get(judge_choice, "auto")

            if st.button("▶️ Start Evaluation Suite", type="primary", width="stretch", key="btn_start_eval_suite"):
                with st.spinner(f"Running evaluation on '{target_agent.name}' ({eval_version})..."):
                    try:
                        # 1. Load dataset based on selection
                        if dataset_choice == "Active Database Test Cases (All Enabled)":
                            eval_dataset = DatasetLoader.from_database(name="database_active", enabled_only=True)
                        elif dataset_choice == "Filter by Tag from Database":
                            eval_dataset = DatasetLoader.from_database(tag=tag_to_run, name=f"tag_{tag_to_run}", enabled_only=True)
                        elif dataset_choice == "golden_tasks (Baseline 15 Tasks)":
                            eval_dataset = DatasetLoader.from_json(DEFAULT_GOLDEN_PATH)
                        else:
                            eval_dataset = DatasetLoader.from_json(dataset_path)

                        if not eval_dataset.test_cases:
                            st.warning("No test cases found in the selected dataset.")
                            st.stop()

                        # 2. Get agent adapter
                        adapter = registry.get_adapter(target_agent.agent_id, version=eval_version)
                        base_agent = adapter.to_base_agent()

                        # 3. Configure Evaluators
                        from src.evaluation.evaluators import (
                            TaskSuccessEvaluator,
                            ToolSelectionEvaluator,
                            KeywordGroundednessEvaluator,
                            LatencyBudgetEvaluator,
                            LLMJudgeEvaluator,
                        )
                        evaluators_to_run = [
                            TaskSuccessEvaluator(),
                            ToolSelectionEvaluator(),
                            KeywordGroundednessEvaluator(),
                            LatencyBudgetEvaluator(),
                            LLMJudgeEvaluator(provider=selected_judge_prov),
                        ]

                        # 4. Initialize EvaluationEngine
                        engine = EvaluationEngine(
                            agent=base_agent,
                            dataset=eval_dataset,
                            metrics=evaluators_to_run,
                            experiment_name=custom_exp_name.strip() if custom_exp_name else None,
                            scoring_config=scoring_config,
                            persist=True,
                        )

                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        total_cases = len(eval_dataset.test_cases)
                        for idx, test_case in enumerate(eval_dataset.test_cases):
                            status_text.text(f"Evaluating {idx+1}/{total_cases}: {test_case.task_id} - {test_case.query[:45]}...")
                            progress_bar.progress((idx + 1) / total_cases)

                        report = engine.run()
                        status_text.text("Evaluation completed successfully!")

                        st.success(f"🎉 Evaluation completed for **{report.experiment_name}**!")

                        # Display Report Summary
                        r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns(5)
                        r_c1.metric("Overall Pass Rate", f"{report.overall_pass_rate:.1f}%")
                        r_c2.metric("Weighted Quality", f"{report.weighted_score:.1f}%")
                        r_c3.metric("Checks Passed", f"{report.passed_checks} / {report.total_checks}")
                        r_c4.metric("Avg Latency", f"{report.avg_latency_ms:.1f} ms")
                        r_c5.metric("Est. Token Cost", f"${report.estimated_cost_usd:.4f}")

                        if report.failure_reasons:
                            with st.expander(f"⚠️ Identified Failure Reasons ({len(set(report.failure_reasons))})", expanded=True):
                                for r_fail in sorted(list(set(report.failure_reasons)))[:5]:
                                    st.markdown(f"- **{r_fail}**")

                        st.markdown("#### Metric Breakdown")
                        st.dataframe(pd.DataFrame(report.metric_breakdown).T, width="stretch")

                        st.cache_data.clear()

                    except Exception as eval_err:
                        st.error(f"Evaluation failed: {str(eval_err)}")
