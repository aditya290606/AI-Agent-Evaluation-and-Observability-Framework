"""
Section: Evaluation Datasets & Non-Blocking Batch Evaluation Suite.

Manages dataset lifecycle, multi-version snapshots, test case membership,
version comparison, and real-time background batch evaluation execution.
"""

import json
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.entities import TestCase, EvaluationDataset, Trace, EvaluationResult
from src.core.dataset_manager import DatasetManager
from src.core.test_case_manager import TestCaseManager
from src.evaluation.batch_runner import BatchEvaluationRunner
from src.evaluation.scoring import ScoringConfig
from src.registry.registry import AgentRegistry
from src.agent.demo_agent import DemoAgent
from src.storage.db import get_session
from src.storage.models import Step
from dashboard.views.common import (
    render_section_header,
    render_kpi_card,
    apply_plotly_theme,
    navigate_to,
    render_failure_analysis,
    steps_to_spans,
)


def render_datasets(
    ds_manager: DatasetManager,
    tc_manager: TestCaseManager,
    batch_runner: BatchEvaluationRunner,
    registry: AgentRegistry,
    scoring_config: ScoringConfig,
):
    # Ensure default dataset is seeded if empty
    ds_manager.seed_default_dataset_if_empty()

    render_section_header(
        title="Evaluation Datasets & Batch Execution",
        subtitle="Manage versioned multi-test benchmark suites, compare dataset compositions, and execute non-blocking batch evaluations.",
        breadcrumb="CORE PLATFORM // DATASETS",
        action_badge="DATASET SUITE",
    )

    # 4 Main Tabs
    tab_catalog, tab_create, tab_compare, tab_runner = st.tabs([
        "📁 Dataset Catalog & Suites",
        "➕ Create New Dataset",
        "⚖️ Compare Dataset Versions",
        "⚡ Batch Evaluation & Runner",
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: Dataset Catalog & Management
    # ---------------------------------------------------------------------------
    with tab_catalog:
        datasets = ds_manager.list_datasets(latest_only=False)

        # Top summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        total_unique_ds = len(set(d.dataset_id for d in datasets))
        total_versions = len(datasets)
        total_tc_refs = sum(len(d.test_cases) for d in datasets)

        with k1:
            render_kpi_card("Total Datasets", str(total_unique_ds), subtitle="Configured Benchmark Suites", icon="📁", accent_color="#38bdf8")
        with k2:
            render_kpi_card("Active Versions", str(total_versions), subtitle="Version Snapshots", icon="🏷️", accent_color="#818cf8")
        with k3:
            render_kpi_card("Total Test Cases", str(total_tc_refs), subtitle="Cumulative Inclusions", icon="🧪", accent_color="#10b981")
        with k4:
            render_kpi_card("Engine Status", "Ready", subtitle="Background Thread Active", health="🟢 Healthy", icon="⚡", accent_color="#10b981")

        st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        if not datasets:
            st.info("No evaluation datasets found. Create your first dataset in the 'Create New Dataset' tab.")
            return

        # Group datasets by dataset_id
        ds_groups: Dict[str, List[EvaluationDataset]] = {}
        for d in datasets:
            if d.dataset_id not in ds_groups:
                ds_groups[d.dataset_id] = []
            ds_groups[d.dataset_id].append(d)

        # Render each dataset family
        for ds_id, v_list in ds_groups.items():
            latest = v_list[0]  # first in descending created_at
            with st.container(border=True):
                # Header row
                hc1, hc2, hc3, hc4 = st.columns([3, 2, 2, 2.5])
                with hc1:
                    st.markdown(
                        f""" <div style="font-size: 16px; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;"> <span>📁</span> {latest.name} </div> <div style="font-size: 11px; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-top: 2px;"> ID: <span style="color: #94a3b8;">{ds_id}</span> </div> """,
                        unsafe_allow_html=True,
                    )
                with hc2:
                    all_vers = [v.version for v in v_list]
                    st.markdown(
                        f""" <div style="font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">VERSIONS ({len(all_vers)})</div> <div style="margin-top: 2px;"> <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.3);">v{latest.version} (latest)</span> </div> """,
                        unsafe_allow_html=True,
                    )
                with hc3:
                    st.markdown(
                        f""" <div style="font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">TEST CASES</div> <div style="font-size: 15px; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace; margin-top: 2px;"> {len(latest.test_cases)} <span style="font-size: 11px; color: #64748b; font-weight: 500;">tests</span> </div> """,
                        unsafe_allow_html=True,
                    )
                with hc4:
                    # Quick run button
                    if st.button(f"🚀 Run v{latest.version}", key=f"btn_quick_run_{ds_id}", width="stretch", type="primary"):
                        st.session_state["runner_preselect_ds"] = ds_id
                        st.session_state["runner_preselect_ver"] = latest.version
                        # Switch to runner tab via session state or rerun
                        st.session_state["ds_active_tab"] = "runner"
                        st.rerun()

                st.markdown(f"<div style='font-size: 12.5px; color: #94a3b8; margin-top: 8px;'>{latest.description or 'No description provided.'}</div>", unsafe_allow_html=True)

                # Expandable details & actions drawer
                with st.expander(f"🔍 Inspect & Manage '{latest.name}' Test Cases ({len(latest.test_cases)} tests)"):
                    # Table of test cases in this dataset
                    tc_rows = []
                    for idx, tc in enumerate(latest.test_cases):
                        diff_color = "#10b981" if tc.difficulty == "easy" else ("#f59e0b" if tc.difficulty == "medium" else "#f43f5e")
                        tc_rows.append({
                            "#": idx + 1,
                            "Test ID": tc.test_id,
                            "Name": tc.name,
                            "Input Query": (tc.user_input[:55] + "...") if len(tc.user_input) > 55 else tc.user_input,
                            "Expected Tools": ", ".join(tc.expected_tools) if tc.expected_tools else "None",
                            "Difficulty": tc.difficulty.upper(),
                            "Budget (ms)": f"{tc.latency_budget:.0f}",
                            "Status": "🟢 Active" if tc.enabled else "⚪ Disabled",
                        })

                    if tc_rows:
                        st.dataframe(pd.DataFrame(tc_rows), width="stretch", hide_index=True)
                    else:
                        st.warning("This dataset currently contains 0 test cases. Add test cases below.")

                    # Action Row: Add / Remove Tests, Duplicate, Version
                    act_c1, act_c2, act_c3 = st.columns(3)

                    # 1. Add / Remove test cases
                    with act_c1:
                        with st.popover("➕ Add / Remove Test Cases"):
                            st.markdown(f"**Manage Test Cases for '{latest.name}' (v{latest.version})**")
                            all_available_tcs = tc_manager.list_test_cases()
                            current_ids = set(tc.test_id for tc in latest.test_cases)
                            all_tc_options = [tc.test_id for tc in all_available_tcs]

                            selected_new_ids = st.multiselect(
                                "Included Test Cases",
                                options=all_tc_options,
                                default=[tid for tid in current_ids if tid in all_tc_options],
                                format_func=lambda tid: f"{tid} — {next((tc.name for tc in all_available_tcs if tc.test_id == tid), tid)}",
                                key=f"manage_tc_multi_{ds_id}_{latest.version}",
                            )

                            if st.button("Save Membership Changes", key=f"btn_save_tc_{ds_id}"):
                                # Determine added and removed
                                selected_set = set(selected_new_ids)
                                to_add = list(selected_set - current_ids)
                                to_remove = list(current_ids - selected_set)

                                if to_add:
                                    ds_manager.add_test_cases(ds_id, to_add, version=latest.version)
                                if to_remove:
                                    ds_manager.remove_test_cases(ds_id, to_remove, version=latest.version)

                                st.success("Updated dataset test case membership!")
                                st.rerun()

                    # 2. Duplicate Dataset
                    with act_c2:
                        with st.popover("📑 Duplicate Dataset"):
                            st.markdown(f"**Clone '{latest.name}'**")
                            clone_name = st.text_input("New Dataset Name", value=f"{latest.name} (Copy)", key=f"dup_name_{ds_id}")
                            clone_id = st.text_input("New Dataset ID", value=f"{ds_id}_copy", key=f"dup_id_{ds_id}")
                            if st.button("Confirm Duplicate", key=f"btn_confirm_dup_{ds_id}", type="primary"):
                                try:
                                    cloned = ds_manager.duplicate_dataset(
                                        dataset_id=ds_id,
                                        new_name=clone_name,
                                        new_dataset_id=clone_id,
                                        source_version=latest.version,
                                    )
                                    st.success(f"Duplicated dataset to '{cloned.name}'!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error duplicating dataset: {str(e)}")

                    # 3. Create New Version
                    with act_c3:
                        with st.popover("🏷️ Create New Version"):
                            st.markdown(f"**Snapshot New Version for '{latest.name}'**")
                            st.caption(f"Current version is v{latest.version}")
                            next_ver_sugg = f"{float(latest.version) + 0.1:.1f}" if latest.version.replace('.', '', 1).isdigit() else f"{latest.version}.1"
                            new_v_tag = st.text_input("New Version Tag*", value=next_ver_sugg, key=f"new_v_tag_{ds_id}")
                            new_v_desc = st.text_input("Version Changelog / Notes", value=f"Updates for v{new_v_tag}", key=f"new_v_desc_{ds_id}")
                            if st.button("Create Version", key=f"btn_confirm_ver_{ds_id}", type="primary"):
                                try:
                                    v_created = ds_manager.version_dataset(
                                        dataset_id=ds_id,
                                        new_version=new_v_tag,
                                        description=new_v_desc,
                                        source_version=latest.version,
                                    )
                                    st.success(f"Created version v{v_created.version} of '{latest.name}'!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to version dataset: {str(e)}")

    # ---------------------------------------------------------------------------
    # TAB 2: Create New Dataset
    # ---------------------------------------------------------------------------
    with tab_create:
        st.subheader("Create a New Evaluation Dataset")
        st.caption("Group test cases into a standardized, reusable evaluation suite with semantic versioning.")

        with st.form("create_dataset_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                new_ds_name = st.text_input("Dataset Name*", placeholder="e.g. Customer Support Inquiries")
                new_ds_desc = st.text_area("Description", placeholder="Comprehensive benchmark evaluating ticket routing, policy search, and discount calculations.")
            with fc2:
                suggested_id = f"ds_{new_ds_name.lower().replace(' ', '_')}" if new_ds_name else "ds_new_suite"
                new_ds_id = st.text_input("Dataset ID*", value=suggested_id, help="Unique slug identifier (e.g. ds_customer_support)")
                new_ds_version = st.text_input("Initial Version", value="1.0")

            st.markdown("#### Select Test Cases to Include")
            all_tcs = tc_manager.list_test_cases()
            if not all_tcs:
                st.warning("No test cases found in the framework database. Create test cases in the 'Test Cases' section first.")

            # Filter controls for picking test cases
            tc_filter_col1, tc_filter_col2 = st.columns(2)
            with tc_filter_col1:
                filter_diff = st.selectbox("Filter by Difficulty", ["All", "Easy", "Medium", "Hard"], key="ds_cr_diff")
            with tc_filter_col2:
                avail_tags = ["All"] + tc_manager.get_all_tags()
                filter_tag = st.selectbox("Filter by Tag", avail_tags, key="ds_cr_tag")

            filtered_pool = all_tcs
            if filter_diff != "All":
                filtered_pool = [c for c in filtered_pool if c.difficulty.lower() == filter_diff.lower()]
            if filter_tag != "All":
                filtered_pool = [c for c in filtered_pool if filter_tag in c.tags]

            selected_tc_ids = st.multiselect(
                "Select Test Cases*",
                options=[c.test_id for c in filtered_pool],
                default=[c.test_id for c in filtered_pool],
                format_func=lambda tid: f"{tid} — {next((c.name for c in filtered_pool if c.test_id == tid), tid)}",
                help="Choose which test cases belong to this dataset."
            )

            submit_create = st.form_submit_button("Create Evaluation Dataset", type="primary")
            if submit_create:
                if not new_ds_name.strip():
                    st.error("Dataset name is required.")
                elif not new_ds_id.strip():
                    st.error("Dataset ID is required.")
                elif not selected_tc_ids:
                    st.error("Please select at least one test case to include in the dataset.")
                else:
                    try:
                        created = ds_manager.create_dataset(
                            name=new_ds_name,
                            description=new_ds_desc,
                            version=new_ds_version,
                            test_case_ids=selected_tc_ids,
                            dataset_id=new_ds_id,
                        )
                        st.success(f"🎉 Created evaluation dataset '{created.name}' (v{created.version}) with {len(created.test_cases)} test cases!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create dataset: {str(e)}")

    # ---------------------------------------------------------------------------
    # TAB 3: Compare Dataset Versions
    # ---------------------------------------------------------------------------
    with tab_compare:
        st.subheader("Compare Dataset Versions")
        st.caption("Inspect compositional differences, test case evolutions, and historical evaluation benchmark performance deltas between versions.")

        all_ds_entities = ds_manager.list_datasets(latest_only=False)
        all_ds_ids = sorted(list(set(d.dataset_id for d in all_ds_entities)))

        if not all_ds_ids:
            st.info("No datasets available to compare.")
            return

        cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
        with cmp_col1:
            target_ds_id = st.selectbox("Select Dataset", all_ds_ids, key="cmp_ds_select")

        versions_available = ds_manager.list_dataset_versions(target_ds_id)
        version_tags = [v.version for v in versions_available]

        with cmp_col2:
            ver_a = st.selectbox("Baseline Version (A)", version_tags, index=0, key="cmp_ver_a")
        with cmp_col3:
            ver_b_idx = min(1, len(version_tags) - 1) if len(version_tags) > 1 else 0
            ver_b = st.selectbox("Target Version (B)", version_tags, index=ver_b_idx, key="cmp_ver_b")

        if ver_a == ver_b:
            st.info("Select two different versions to compute compositional and benchmark deltas.")
            return

        diff_res = ds_manager.compare_dataset_versions(target_ds_id, ver_a, ver_b)

        # 1. Summary Scorecards
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            render_kpi_card(f"Tests in v{ver_a}", str(diff_res["version_a"]["total_tests"]), subtitle="Baseline Suite Count", icon="🔢", accent_color="#94a3b8")
        with m_c2:
            render_kpi_card(f"Tests in v{ver_b}", str(diff_res["version_b"]["total_tests"]), subtitle="Target Suite Count", icon="🔢", accent_color="#38bdf8")
        with m_c3:
            render_kpi_card("Added Tests", f"+{len(diff_res['added_test_cases'])}", subtitle=f"New in v{ver_b}", icon="➕", accent_color="#10b981")
        with m_c4:
            render_kpi_card("Removed Tests", f"-{len(diff_res['removed_test_cases'])}", subtitle=f"Excluded from v{ver_b}", icon="➖", accent_color="#f43f5e")

        st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # 2. Added & Removed Test Case Breakdowns
        diff_tab1, diff_tab2, diff_tab3 = st.tabs(["➕ Added Test Cases", "➖ Removed Test Cases", "📊 Evaluation Quality Delta"])

        with diff_tab1:
            if diff_res["added_test_cases"]:
                st.markdown(f"**{len(diff_res['added_test_cases'])} Test Cases Added in v{ver_b}:**")
                added_df = pd.DataFrame([{
                    "Test ID": tc["test_id"],
                    "Name": tc["name"],
                    "User Input Query": tc["user_input"],
                    "Expected Tools": ", ".join(tc.get("expected_tools", [])),
                    "Difficulty": tc.get("difficulty", "medium").upper(),
                } for tc in diff_res["added_test_cases"]])
                st.dataframe(added_df, width="stretch", hide_index=True)
            else:
                st.info(f"Zero test cases were added in v{ver_b}.")

        with diff_tab2:
            if diff_res["removed_test_cases"]:
                st.markdown(f"**{len(diff_res['removed_test_cases'])} Test Cases Removed from v{ver_a}:**")
                rem_df = pd.DataFrame([{
                    "Test ID": tc["test_id"],
                    "Name": tc["name"],
                    "User Input Query": tc["user_input"],
                    "Expected Tools": ", ".join(tc.get("expected_tools", [])),
                    "Difficulty": tc.get("difficulty", "medium").upper(),
                } for tc in diff_res["removed_test_cases"]])
                st.dataframe(rem_df, width="stretch", hide_index=True)
            else:
                st.info(f"Zero test cases were removed in v{ver_b}.")

        with diff_tab3:
            eval_diff = diff_res.get("evaluation_diff")
            if eval_diff and eval_diff.get("has_benchmark_data"):
                st.markdown("#### Historical Evaluation Performance Delta")
                ed_c1, ed_c2, ed_c3 = st.columns(3)
                with ed_c1:
                    pass_delta = eval_diff["pass_rate_delta"]
                    render_kpi_card(
                        "Pass Rate Delta",
                        f"{pass_delta:+.1f}%",
                        subtitle=f"v{ver_a}: {eval_diff['metrics_a']['pass_rate_pct']}% ➔ v{ver_b}: {eval_diff['metrics_b']['pass_rate_pct']}%",
                        health="🟢 Healthy" if pass_delta >= 0 else "🔴 Critical",
                        icon="📈",
                        accent_color="#10b981" if pass_delta >= 0 else "#f43f5e"
                    )
                with ed_c2:
                    lat_delta = eval_diff["latency_delta_ms"]
                    render_kpi_card(
                        "Latency Delta",
                        f"{lat_delta:+.1f} ms",
                        subtitle=f"v{ver_a}: {eval_diff['metrics_a']['avg_latency_ms']:.0f}ms ➔ v{ver_b}: {eval_diff['metrics_b']['avg_latency_ms']:.0f}ms",
                        icon="⏱️",
                        accent_color="#38bdf8"
                    )
                with ed_c3:
                    cost_delta = eval_diff["cost_delta_usd"]
                    render_kpi_card(
                        "Cost Delta",
                        f"${cost_delta:+.4f}",
                        subtitle=f"v{ver_a}: ${eval_diff['metrics_a']['total_cost_usd']:.4f} ➔ v{ver_b}: ${eval_diff['metrics_b']['total_cost_usd']:.4f}",
                        icon="🪙",
                        accent_color="#f59e0b"
                    )

                # Comparison Chart
                cmp_chart_df = pd.DataFrame([
                    {"Version": f"v{ver_a}", "Pass Rate (%)": eval_diff['metrics_a']['pass_rate_pct'], "Avg Latency (ms)": eval_diff['metrics_a']['avg_latency_ms']},
                    {"Version": f"v{ver_b}", "Pass Rate (%)": eval_diff['metrics_b']['pass_rate_pct'], "Avg Latency (ms)": eval_diff['metrics_b']['avg_latency_ms']},
                ])
                fig_cmp = px.bar(
                    cmp_chart_df,
                    x="Version",
                    y="Pass Rate (%)",
                    color="Version",
                    text_auto=".1f",
                    color_discrete_sequence=["#94a3b8", "#38bdf8"]
                )
                apply_plotly_theme(fig_cmp, height=260)
                st.plotly_chart(fig_cmp, width="stretch")
            else:
                st.info(f"No historical evaluation runs found logged under both v{ver_a} and v{ver_b}. Run evaluations for both versions in the 'Batch Evaluation' tab to see quality deltas.")

    # ---------------------------------------------------------------------------
    # TAB 4: Batch Evaluation & Runner
    # ---------------------------------------------------------------------------
    with tab_runner:
        st.subheader("⚡ Non-Blocking Batch Evaluation Runner")
        st.caption("Trigger full suite executions across AI agents. The evaluation runs in the background without blocking the UI.")

        # Pre-selections from session state if any
        pre_ds = st.session_state.pop("runner_preselect_ds", None)
        pre_ver = st.session_state.pop("runner_preselect_ver", None)

        all_ds_entries = ds_manager.list_datasets(latest_only=False)
        ds_id_options = sorted(list(set(d.dataset_id for d in all_ds_entries)))

        if not ds_id_options:
            st.info("No datasets available. Create a dataset first.")
            return

        run_cfg_c1, run_cfg_c2, run_cfg_c3, run_cfg_c4 = st.columns(4)
        with run_cfg_c1:
            sel_ds_id = st.selectbox("Dataset Suite", ds_id_options, index=ds_id_options.index(pre_ds) if pre_ds in ds_id_options else 0, key="batch_sel_ds")
        with run_cfg_c2:
            ds_vers = [v.version for v in ds_manager.list_dataset_versions(sel_ds_id)]
            sel_ver = st.selectbox("Version", ds_vers, index=ds_vers.index(pre_ver) if pre_ver in ds_vers else 0, key="batch_sel_ver")
        with run_cfg_c3:
            # Agents from registry
            registered_agents = registry.list_agents()
            agent_names = [a.name for a in registered_agents] if registered_agents else ["DemoAgent"]
            sel_agent_name = st.selectbox("Target Agent", agent_names, index=0, key="batch_sel_agent")
        with run_cfg_c4:
            agent_obj = registry.get_agent(sel_agent_name)
            default_model = agent_obj.provider_model if agent_obj else "claude-3-5-haiku"
            sel_model = st.selectbox("Model Provider", ["claude-3-5-haiku", "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"], index=0, key="batch_sel_model")

        target_dataset = ds_manager.get_dataset(sel_ds_id, version=sel_ver)
        if not target_dataset:
            st.error("Failed to load selected dataset.")
            return

        st.markdown(
            f""" <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;"> <div> <span style="font-weight: 700; color: #f8fafc;">{target_dataset.name}</span> <span style="color: #64748b; font-size: 12px; margin-left: 8px;">(Version {target_dataset.version} · {len(target_dataset.test_cases)} tests)</span> </div> <div style="font-size: 12px; color: #38bdf8; font-family: 'JetBrains Mono', monospace;"> ⚡ Non-Blocking Execution Thread </div> </div> """,
            unsafe_allow_html=True,
        )

        # Trigger button
        trig_c1, trig_c2 = st.columns([2, 5])
        with trig_c1:
            start_btn = st.button("🚀 Run Dataset Evaluation", type="primary", width="stretch", key="btn_start_batch")

        if start_btn:
            # Instantiate agent
            active_agent = None
            if agent_obj:
                try:
                    active_agent = registry.get_agent_instance(agent_obj.name)
                except Exception:
                    active_agent = DemoAgent(use_mock=True, model_name=sel_model)
            else:
                active_agent = DemoAgent(use_mock=True, model_name=sel_model)

            # Start background evaluation
            job_id = batch_runner.start_batch_run(
                agent=active_agent,
                dataset=target_dataset,
                scoring_config=scoring_config,
                model=sel_model,
            )
            st.session_state["active_batch_job_id"] = job_id
            st.success(f"Started evaluation run `{job_id}` in background thread!")
            st.rerun()

        # Check for active or latest job
        active_job_id = st.session_state.get("active_batch_job_id")
        latest_job = batch_runner.get_job(active_job_id) if active_job_id else batch_runner.get_latest_job()

        if latest_job:
            st.markdown("<div style='margin-top: 14px; margin-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

            # -------------------------------------------------------------------
            # Progress Display (Non-Blocking)
            # -------------------------------------------------------------------
            if latest_job.status in ["pending", "running"]:
                pct = max(0.0, min(1.0, latest_job.progress_pct / 100.0))
                st.progress(pct)

                p_c1, p_c2 = st.columns([4, 1])
                with p_c1:
                    curr_str = f" • Current: {latest_job.current_test_id}" if latest_job.current_test_id else ""
                    st.markdown(
                        f""" <div style="font-size: 13px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px;"> <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#38bdf8; box-shadow:0 0 8px #38bdf8;"></span> Running test {latest_job.completed_tests + 1} of {latest_job.total_tests} ({latest_job.progress_pct:.0f}%){curr_str} <span style="color: #64748b; font-size: 11.5px; font-weight: 500;">(Elapsed: {latest_job.elapsed_seconds:.1f}s)</span> </div> """,
                        unsafe_allow_html=True,
                    )
                with p_c2:
                    if st.button("🛑 Cancel Run", key=f"btn_cancel_{latest_job.job_id}"):
                        batch_runner.cancel_job(latest_job.job_id)
                        st.warning("Cancellation requested...")
                        st.rerun()

                # Seamless auto-polling using Streamlit sleep + rerun
                time.sleep(1.2)
                st.rerun()

            elif latest_job.status == "cancelled":
                st.warning(f"Evaluation run `{latest_job.job_id}` was cancelled by user. ({latest_job.completed_tests} of {latest_job.total_tests} tests executed)")

            elif latest_job.status == "failed":
                st.error(f"Evaluation run `{latest_job.job_id}` failed: {latest_job.error_message}")

            elif latest_job.status == "completed":
                summary = latest_job.summary or batch_runner.get_evaluation_run_summary(latest_job.experiment_id)

                st.markdown(
                    f""" <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;"> <div> <div style="font-size: 11px; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.1em;">BATCH EXECUTION COMPLETED</div> <div style="font-size: 20px; font-weight: 800; color: #f8fafc;"> Evaluation Run: {summary.get('dataset_name', 'Dataset')} ({summary.get('dataset_version', '1.0')}) </div> </div> <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; font-size: 11.5px; padding: 4px 12px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3);"> ● RUN #{summary.get('experiment_id', 'run')[:16]} </span> </div> """,
                    unsafe_allow_html=True,
                )

                # Row 1 of Scorecard: Test Counts & Overall Score
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                tot = summary.get("total_tests", 0)
                pas = summary.get("passed", 0)
                fai = summary.get("failed", 0)
                ski = latest_job.skipped_count

                with sc1:
                    render_kpi_card("Total Tests", str(tot), subtitle="Suite Execution", icon="🔢", accent_color="#38bdf8")
                with sc2:
                    p_pct = (pas / tot * 100) if tot else 0.0
                    render_kpi_card("Passed", f"{pas} ({p_pct:.0f}%)", health="🟢 Healthy" if p_pct >= 90 else "🟡 Degraded", icon="✅", accent_color="#10b981")
                with sc3:
                    f_pct = (fai / tot * 100) if tot else 0.0
                    render_kpi_card("Failed", f"{fai} ({f_pct:.0f}%)", health="🔴 Critical" if fai > 0 else "🟢 Healthy", icon="❌", accent_color="#f43f5e")
                with sc4:
                    render_kpi_card("Skipped", str(ski), subtitle="Disabled Test Cases", icon="⚪", accent_color="#64748b")
                with sc5:
                    ov_score = summary.get("overall_score", 0.0)
                    render_kpi_card("Overall Score", f"{ov_score:.1f}%", subtitle="Weighted Suite Score", health="🟢 Healthy" if ov_score >= 85 else "🟡 Degraded", icon="🎯", accent_color="#818cf8")

                # Row 2 of Scorecard: Latency, Cost, Tokens
                sc6, sc7, sc8, sc9 = st.columns(4)
                with sc6:
                    render_kpi_card("Avg Latency", f"{summary.get('average_latency_ms', 0):.0f} ms", subtitle="Mean Duration", icon="⏱️", accent_color="#38bdf8")
                with sc7:
                    render_kpi_card("P95 Latency", f"{summary.get('p95_latency_ms', 0):.0f} ms", subtitle="95th Percentile", icon="⚡", accent_color="#f59e0b")
                with sc8:
                    render_kpi_card("Total Cost", f"${summary.get('total_cost_usd', 0):.4f}", subtitle="Model + Tool Invocations", icon="🪙", accent_color="#10b981")
                with sc9:
                    render_kpi_card("Total Tokens", f"{summary.get('total_tokens', 0):,}", subtitle="Input + Output Tokens", icon="🔢", accent_color="#a855f7")

                # Metric Scores Breakdown
                metric_scores = summary.get("metric_scores", {})
                if metric_scores:
                    st.markdown("#### Metric Scores Breakdown")
                    m_cols = st.columns(len(metric_scores))
                    for idx, (m_name, m_data) in enumerate(metric_scores.items()):
                        with m_cols[idx]:
                            m_title = m_name.replace("_", " ").title()
                            pr = m_data.get("pass_rate_pct", 0)
                            render_kpi_card(
                                m_title,
                                f"{pr:.0f}%",
                                subtitle=f"{m_data.get('passed', 0)}/{m_data.get('total', 0)} Passed · Avg: {m_data.get('avg_score', 0):.2f}",
                                health="🟢 Healthy" if pr >= 85 else "🔴 Critical",
                                accent_color="#38bdf8"
                            )

                st.markdown("<div style='margin-top: 14px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

                # Individual Test Executions List (with filter & drill-downs)
                st.markdown(
                    """ <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;"> <div style="font-size: 16px; font-weight: 800; color: #f8fafc;"> Individual Test Case Executions & Telemetry </div> <div style="font-size: 11px; color: #64748b; font-family: 'JetBrains Mono', monospace;"> CLICK ANY FAILED TEST TO OPEN TRACE & ROOT CAUSE (RCA) </div> </div> """,
                    unsafe_allow_html=True,
                )

                test_filter = st.radio("Filter Tests", ["All Tests", "Failed Only", "Passed Only"], horizontal=True, key="ds_test_view_filter")

                test_runs = summary.get("test_runs", [])
                if test_filter == "Failed Only":
                    test_runs = [t for t in test_runs if not t.get("passed", True)]
                elif test_filter == "Passed Only":
                    test_runs = [t for t in test_runs if t.get("passed", True)]

                for tr in test_runs:
                    rid = tr["run_id"]
                    is_p = tr.get("passed", False)
                    status_pill = (
                        '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#10b981;"></span>PASSED</span>'
                        if is_p else
                        '<span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 9999px; border: 1px solid rgba(244, 63, 94, 0.3); display: inline-flex; align-items: center; gap: 4px;"><span style="width:5px; height:5px; border-radius:50%; background:#f43f5e;"></span>FAILED</span>'
                    )

                    with st.container(border=True):
                        tc1, tc2, tc3, tc4, tc5 = st.columns([1.5, 3.5, 1.2, 1.2, 2.6])
                        with tc1:
                            st.markdown(f"<span style='font-family: \"JetBrains Mono\", monospace; font-weight: 700; color: #f8fafc;'>Run #{rid}</span><br>{status_pill}", unsafe_allow_html=True)
                        with tc2:
                            q_snip = (tr['query'][:55] + "...") if len(tr['query']) > 55 else tr['query']
                            st.markdown(
                                f""" <span style="color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{tr['task_id']}</span> <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">{q_snip}</div> """,
                                unsafe_allow_html=True,
                            )
                        with tc3:
                            st.markdown(f"⏱️ **{tr['latency_ms']:.0f} ms**<br><span style='font-size: 11px; color: #64748b;'>Latency</span>", unsafe_allow_html=True)
                        with tc4:
                            st.markdown(f"🎯 **{tr['score']:.0f}%**<br><span style='font-size: 11px; color: #64748b;'>Score</span>", unsafe_allow_html=True)
                        with tc5:
                            # Action buttons (Trace & RCA drill downs)
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("🔍 Trace ➔", key=f"ds_btn_tr_{rid}", help="Inspect span timeline and tokens"):
                                    navigate_to("🔍 Traces", selected_run_id=rid)
                            with btn_col2:
                                if not is_p:
                                    if st.button("🚨 RCA ➔", key=f"ds_btn_rca_{rid}", help="AI Root Cause Analysis"):
                                        navigate_to("🚨 Failures", selected_run_id=rid)
                                else:
                                    st.caption("✅ Passed")

                    if not is_p:
                        with st.expander("🔬 Failure Analysis", expanded=True):
                            t_case = tc_manager.get_test_case(tr.get("task_id")) if tc_manager else None
                            session_db = get_session()
                            try:
                                step_rows = session_db.query(Step).filter(Step.run_id == rid).order_by(Step.step_index.asc()).all()
                                span_objects = steps_to_spans(step_rows) if step_rows else []
                            except Exception:
                                span_objects = []
                            finally:
                                session_db.close()

                            trace_obj = Trace(
                                task_id=tr.get("task_id", ""),
                                query=tr.get("query", ""),
                                final_answer=tr.get("final_answer", ""),
                                spans=span_objects,
                            )
                            trace_obj.latency_ms = float(tr.get("latency_ms") or 0.0)
                            if not span_objects and tr.get("tools_called"):
                                trace_obj.metadata["tools_called"] = tr.get("tools_called")

                            eval_objs = []
                            for ed in tr.get("evaluations", []):
                                eval_objs.append(EvaluationResult(
                                    metric_name=ed["metric_name"],
                                    score=float(ed.get("score", 0.0) or 0.0),
                                    passed=bool(ed.get("passed", False)),
                                    threshold=1.0,
                                    explanation=ed.get("details", "") or "",
                                ))
                            render_failure_analysis(t_case, trace_obj, eval_objs, show_all=True)
