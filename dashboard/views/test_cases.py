"""
Section 3: Test Cases - Configurable Evaluation Test Case System.

Provides:
- 8-Field Schema: test_id, name, user_input, expected_behavior, expected_tool, expected_answer/keywords, latency_budget, enabled
- Full lifecycle: create, edit, enable/disable, run single test, run all tests (suite)
- Preserves all 15 baseline test cases (T001 - T015)
- Seamless compatibility with existing EvaluationEngine
"""

import json
from typing import Any, List
import pandas as pd
import streamlit as st

from src.core.test_case_manager import TestCaseManager
from src.registry.registry import AgentRegistry
from src.evaluation.scoring import calculate_case_scores
from dashboard.views.common import render_section_header, render_kpi_card, render_failure_analysis


def render_test_cases(tc_manager: TestCaseManager, registry: AgentRegistry):
    render_section_header(
        title="Evaluation Test System",
        subtitle="Configure, manage, and execute evaluation tests across your agent portfolio.",
        breadcrumb="BENCHMARK // TEST SUITE",
        action_badge="CONFIGURABLE TESTS",
    )

    # Ensure baseline test cases are synchronized
    all_test_cases = tc_manager.list_test_cases()
    enabled_cases = [t for t in all_test_cases if t.enabled]
    disabled_cases = [t for t in all_test_cases if not t.enabled]
    baseline_count = sum(1 for t in all_test_cases if t.test_id.startswith("T0"))

    # Top KPI Metrics
    tc_kpi1, tc_kpi2, tc_kpi3, tc_kpi4 = st.columns(4)
    with tc_kpi1:
        render_kpi_card("Total Tests", str(len(all_test_cases)), subtitle="Registered Test Cases", icon="🧪", accent_color="#38bdf8")
    with tc_kpi2:
        render_kpi_card("Enabled", str(len(enabled_cases)), subtitle="Active in Test Suites", icon="🟢", accent_color="#10b981")
    with tc_kpi3:
        render_kpi_card("Disabled", str(len(disabled_cases)), subtitle="Excluded / Archived", icon="⚪", accent_color="#64748b")
    with tc_kpi4:
        render_kpi_card("Baseline Tests", f"{baseline_count}/15", subtitle="Preserved (T001–T015)", icon="🛡️", accent_color="#8b5cf6")

    st.markdown("<div style='margin-top: 14px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SUITE RUNNER: Run All Tests Action Bar
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            """ <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;"> <div> <span style="font-size: 15px; font-weight: 700; color: #f8fafc;">🚀 Run Evaluation Test Suite</span> <span style="font-size: 12px; color: #94a3b8; margin-left: 8px;">Execute test suite against an active agent</span> </div> <span style="font-size: 12px; color: #38bdf8; font-family: 'JetBrains Mono', monospace;"> Target: %d Enabled Tests </span> </div> """ % len(enabled_cases),
            unsafe_allow_html=True,
        )

        active_agents = registry.list_agents(status="active")
        if not active_agents:
            st.warning("No active agents found in registry.")
        else:
            s_col1, s_col2, s_col3 = st.columns([3, 2, 2])
            with s_col1:
                agent_names = [a.name for a in active_agents]
                suite_agent_name = st.selectbox("Select Agent", agent_names, key="suite_agent_select")
                selected_agent_obj = registry.get_agent(suite_agent_name)
            with s_col2:
                v_choices = [v.version for v in registry.get_agent_versions(selected_agent_obj.agent_id)] if selected_agent_obj else ["v1.0"]
                suite_version = st.selectbox("Version", v_choices, key="suite_version_select")
            with s_col3:
                st.write("")
                st.write("")
                run_all_btn = st.button("🚀 Run All Tests", key="btn_run_all_suite", type="primary", width="stretch")

            if run_all_btn:
                if not enabled_cases:
                    st.error("Cannot run test suite: No tests are currently enabled. Enable at least one test below.")
                else:
                    with st.spinner(f"Running test suite ({len(enabled_cases)} tests) against {suite_agent_name} ({suite_version})..."):
                        try:
                            adapter = registry.get_adapter(selected_agent_obj.agent_id, version=suite_version)
                            base_agent = adapter.to_base_agent()
                            report, summaries = tc_manager.run_all_tests(base_agent, enabled_only=True)

                            st.session_state["suite_last_report"] = report
                            st.session_state["suite_last_summaries"] = summaries
                            st.session_state["suite_last_agent"] = f"{suite_agent_name} ({suite_version})"
                            st.success(f"✅ Test Suite Completed! Executed {report.total_test_cases} test cases.")
                        except Exception as ex:
                            st.error(f"Suite execution error: {str(ex)}")

        # Display Suite Results if available in session
        if "suite_last_report" in st.session_state:
            rep = st.session_state["suite_last_report"]
            sums = st.session_state["suite_last_summaries"]
            tgt = st.session_state.get("suite_last_agent", "Agent")

            st.markdown("<div style='margin-top: 12px; margin-bottom: 12px; border-top: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)
            st.markdown(f"##### 📊 Suite Execution Results: `{tgt}`")

            # Metrics row
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            with res_c1:
                pass_color = "#10b981" if rep.overall_pass_rate >= 80 else ("#f59e0b" if rep.overall_pass_rate >= 50 else "#f43f5e")
                render_kpi_card("Overall Pass Rate", f"{rep.overall_pass_rate:.1f}%", subtitle="Checks Passed", icon="📈", accent_color=pass_color)
            with res_c2:
                passed_cnt = sum(1 for s in sums if s["passed"])
                render_kpi_card("Passed Tests", f"{passed_cnt}/{len(sums)}", subtitle="Tests Satisfied", icon="✅", accent_color="#10b981")
            with res_c3:
                failed_cnt = len(sums) - passed_cnt
                render_kpi_card("Failed Tests", str(failed_cnt), subtitle="Needs Review", icon="❌" if failed_cnt > 0 else "🎉", accent_color="#f43f5e" if failed_cnt > 0 else "#10b981")
            with res_c4:
                render_kpi_card("Avg Latency", f"{rep.avg_latency_ms:.0f} ms", subtitle="Per Test Case", icon="⚡", accent_color="#38bdf8")

            # Expandable Breakdown Table
            with st.expander("📋 View Detailed Per-Test Suite Verdicts", expanded=True):
                table_rows = []
                for s in sums:
                    table_rows.append({
                        "Status": "✅ PASS" if s["passed"] else "❌ FAIL",
                        "Test ID": s["test_id"],
                        "Name": s["name"],
                        "Expected Tool": s["expected_tool"] or "None",
                        "Budget": f"{s['latency_budget']:.0f}ms",
                        "Composite Score": f"{s['score']:.2f}",
                        "Verdict Reason": "; ".join(s["failure_reasons"]) if s["failure_reasons"] else "All evaluation checks passed",
                    })
                df_results = pd.DataFrame(table_rows)
                st.dataframe(df_results, width="stretch", hide_index=True)

            failed_sums = [s for s in sums if not s.get("passed", True)]
            if failed_sums:
                with st.expander(f"🔬 Failure Analysis for Suite Failures ({len(failed_sums)})", expanded=True):
                    st.caption("Automated classification of failed suite tests separating observed evidence from inferred root causes.")
                    for fs in failed_sums:
                        ftc = tc_manager.get_test_case(fs["test_id"])
                        from src.core.entities import Trace as CoreTrace, EvaluationResult as CoreEvalResult
                        eval_objs = []
                        for m_name, m_score in fs.get("individual_scores", {}).items():
                            eval_objs.append(CoreEvalResult(
                                metric_name=m_name,
                                score=float(m_score),
                                passed=float(m_score) >= 0.8,
                                threshold=0.8,
                                explanation="; ".join(fs.get("failure_reasons", [])) if float(m_score) < 0.8 else "Passed",
                            ))
                        f_trace = CoreTrace(
                            task_id=fs["test_id"],
                            query=fs.get("user_input", ""),
                            final_answer=fs.get("final_answer", ""),
                            spans=[],
                        )
                        f_trace.latency_ms = float(fs.get("latency_budget", 5000.0) + (100.0 if "latency" in str(fs.get("failure_reasons", "")).lower() else 0.0))
                        render_failure_analysis(ftc, f_trace, eval_objs, show_all=True)

    st.markdown("<div style='margin-top: 20px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # CREATE NEW TEST CASE (8-Field Schema)
    # -----------------------------------------------------------------------
    with st.expander("➕ Create New Test Case", expanded=False):
        st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-bottom: 12px;'>Add a new evaluation scenario with standard test specifications.</p>", unsafe_allow_html=True)
        with st.form("create_test_form"):
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                f_test_id = st.text_input("1. Test ID*", placeholder="e.g. TC_016 or TC_REFUND_01")
                f_name = st.text_input("2. Name*", placeholder="e.g. Express Shipping Rates")
                f_user_input = st.text_area("3. User Input (Prompt / Query)*", placeholder="How much extra does express shipping cost?", height=85)
            with c_col2:
                f_behavior = st.text_area("4. Expected Behavior*", placeholder="Query knowledge base for express shipping surcharge and retrieve 499 INR", height=85)
                f_tool = st.text_input("5. Expected Tool (Optional)", placeholder="e.g. search_knowledge_base, calculator, or leave empty")
                f_answer_kws = st.text_input("6. Expected Answer / Keywords (Optional)", placeholder="e.g. 499, express")

            c_col3, c_col4 = st.columns(2)
            with c_col3:
                f_budget = st.number_input("7. Latency Budget (ms)*", min_value=100.0, max_value=30000.0, value=5000.0, step=500.0)
            with c_col4:
                st.write("")
                f_enabled = st.checkbox("8. Enabled (Include in test runs)", value=True)

            create_submit = st.form_submit_button("Create Test Case", type="primary", width="stretch")
            if create_submit:
                if not f_test_id.strip() or not f_name.strip() or not f_user_input.strip() or not f_behavior.strip():
                    st.error("Please fill in all required fields: Test ID, Name, User Input, and Expected Behavior.")
                else:
                    try:
                        kws_list = [k.strip() for k in f_answer_kws.split(",") if k.strip()] if f_answer_kws.strip() else []
                        created = tc_manager.create_test_case({
                            "test_id": f_test_id.strip(),
                            "name": f_name.strip(),
                            "user_input": f_user_input.strip(),
                            "expected_behavior": f_behavior.strip(),
                            "expected_tool": f_tool.strip() if f_tool.strip() else None,
                            "expected_keywords": kws_list,
                            "expected_answer": f_answer_kws.strip() if f_answer_kws.strip() else None,
                            "latency_budget": float(f_budget),
                            "enabled": f_enabled,
                        })
                        st.success(f"✅ Created test case '{created.test_id}' ({created.name})!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create test case: {str(e)}")

    st.markdown("<div style='margin-top: 15px; margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # CATALOG & FILTER CONTROLS
    # -----------------------------------------------------------------------
    fc1, fc2, fc3 = st.columns([4, 2, 2])
    with fc1:
        search_query = st.text_input("🔍 Search Test Cases", placeholder="Search by Test ID, name, or query...", key="tc_cat_search")
    with fc2:
        status_choice = st.selectbox("Status Filter", ["All", "Enabled Only", "Disabled Only"], key="tc_cat_status")
    with fc3:
        st.write("")
        if st.button("🔄 Sync Baseline (T001–T015)", key="btn_sync_baseline", help="Re-sync baseline test cases from golden_tasks.json"):
            synced = tc_manager.sync_baseline_tests(force_overwrite=True)
            st.success(f"Synchronized {synced} baseline test cases!")
            st.rerun()

    # Filter test cases
    filtered_tests = tc_manager.list_test_cases(
        enabled_only=(status_choice == "Enabled Only"),
        search=search_query.strip() if search_query else None,
    )
    if status_choice == "Disabled Only":
        filtered_tests = [t for t in filtered_tests if not t.enabled]

    st.markdown(
        """ <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 12px;"> <span style="font-size: 15px; font-weight: 700; color: #f8fafc;">Test Catalog (%d Tests)</span> <span style="font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Configured Evaluation Cases</span> </div> """ % len(filtered_tests),
        unsafe_allow_html=True,
    )

    if not filtered_tests:
        st.info("No test cases match the active filter criteria.")
    else:
        # Render cards
        display_cases = filtered_tests[:15]
        for t in display_cases:
            status_color = "#10b981" if t.enabled else "#64748b"
            status_label = "ENABLED" if t.enabled else "DISABLED"

            with st.container(border=True):
                card_c1, card_c2, card_c3 = st.columns([5, 3.5, 2.5])
                with card_c1:
                    st.markdown(
                        f""" <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;"> <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 7px; border-radius: 4px;">{t.test_id}</span> <span style="font-size: 13.5px; font-weight: 700; color: #f8fafc;">{t.name}</span> </div> <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 4px; line-height: 1.4;"> <strong style="color: #94a3b8;">Input:</strong> <em>"{t.user_input}"</em> </div> <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.3;"> <strong style="color: #64748b;">Expected Behavior:</strong> {t.expected_behavior} </div> """,
                        unsafe_allow_html=True,
                    )
                with card_c2:
                    tool_display = t.expected_tool if t.expected_tool else "none"
                    kws_display = ", ".join(t.expected_keywords) if t.expected_keywords else (t.expected_answer or "none")
                    st.markdown(
                        f""" <div style="font-size: 11px; color: #94a3b8; margin-bottom: 3px;">EXPECTED TOOL: <code style="color: #a78bfa; font-size: 10.5px;">{tool_display}</code></div> <div style="font-size: 11px; color: #94a3b8; margin-bottom: 3px;">KEYWORDS: <span style="color: #cbd5e1; font-size: 10.5px;">{kws_display}</span></div> <div style="font-size: 11px; color: #94a3b8;">LATENCY BUDGET: <span style="color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 10.5px;">{t.latency_budget:.0f}ms</span></div> """,
                        unsafe_allow_html=True,
                    )
                with card_c3:
                    st.markdown(
                        f""" <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;"> <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 9999px; background: {status_color}18; border: 1px solid {status_color}40; color: {status_color}; font-size: 10.5px; font-weight: 700;"> <span style="width: 5px; height: 5px; border-radius: 50%; background: {status_color};"></span>{status_label} </span> </div> """,
                        unsafe_allow_html=True,
                    )
                    # Quick action buttons
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if t.enabled:
                            if st.button("Disable", key=f"btn_dis_{t.test_id}", width="stretch"):
                                tc_manager.toggle_test_case(t.test_id, False)
                                st.rerun()
                        else:
                            if st.button("Enable", key=f"btn_en_{t.test_id}", width="stretch"):
                                tc_manager.toggle_test_case(t.test_id, True)
                                st.rerun()
                    with b_col2:
                        if st.button("Delete", key=f"btn_del_{t.test_id}", width="stretch"):
                            tc_manager.delete_test_case(t.test_id)
                            st.warning(f"Deleted test case '{t.test_id}'.")
                            st.rerun()

        if len(filtered_tests) > 15:
            st.caption(f"Showing 15 of {len(filtered_tests)} matching test cases. Use search bar above to narrow down.")

        st.markdown("<div style='margin-top: 20px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # TEST INSPECTION, RUN SINGLE TEST & EDIT
        # -------------------------------------------------------------------
        st.subheader("🛠️ Inspect, Run & Edit Test Case")
        selected_id = st.selectbox("Select Test Case to Inspect / Run / Edit", [t.test_id for t in filtered_tests], key="inspect_tc_select")
        active_tc = tc_manager.get_test_case(selected_id)

        if active_tc:
            with st.container(border=True):
                id_c1, id_c2 = st.columns(2)
                with id_c1:
                    st.markdown(f"**Test ID:** `{active_tc.test_id}`")
                    st.markdown(f"**Name:** {active_tc.name}")
                    st.markdown(f"**User Input:** {active_tc.user_input}")
                    st.markdown(f"**Expected Behavior:** {active_tc.expected_behavior}")
                with id_c2:
                    st.markdown(f"**Expected Tool:** `{active_tc.expected_tool or 'None'}`")
                    kws_str = ", ".join(active_tc.expected_keywords) if active_tc.expected_keywords else (active_tc.expected_answer or "None")
                    st.markdown(f"**Expected Keywords / Answer:** `{kws_str}`")
                    st.markdown(f"**Latency Budget:** `{active_tc.latency_budget:.0f} ms`")
                    st.markdown(f"**Status:** {'🟢 Enabled' if active_tc.enabled else '⚪ Disabled'}")

            # Tabbed Operations: [1] Run Single Test, [2] Edit Test Case
            tab_run, tab_edit = st.tabs(["⚡ Run Single Test", "✏️ Edit Test Case"])

            with tab_run:
                run_agents = registry.list_agents(status="active")
                if run_agents:
                    ra_c1, ra_c2 = st.columns([3, 2])
                    with ra_c1:
                        chosen_agent_name = st.selectbox("Agent to Evaluate", [a.name for a in run_agents], key="single_run_agent")
                        chosen_agent_obj = registry.get_agent(chosen_agent_name)
                        ver_opts = [v.version for v in registry.get_agent_versions(chosen_agent_obj.agent_id)] if chosen_agent_obj else ["v1.0"]
                        chosen_ver = st.selectbox("Version", ver_opts, key="single_run_version")
                    with ra_c2:
                        st.write("")
                        st.write("")
                        single_exec_btn = st.button("⚡ Execute Test", key="btn_exec_single_test", type="primary", width="stretch")

                    if single_exec_btn:
                        with st.spinner(f"Running '{active_tc.test_id}' against {chosen_agent_name} ({chosen_ver})..."):
                            try:
                                adapter = registry.get_adapter(chosen_agent_obj.agent_id, version=chosen_ver)
                                base_agent = adapter.to_base_agent()
                                trace, eval_results = tc_manager.run_single_test(base_agent, active_tc.test_id)

                                st.success("Execution and Evaluation completed!")
                                st.markdown(f"**Agent Final Answer:** {trace.final_answer}")

                                c_summary = calculate_case_scores(eval_results)
                                if trace.is_mock or c_summary.get("is_mock"):
                                    st.warning("⚠️ **SIMULATED RUN**: Mock scores must never be presented as real evaluation results.")

                                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                                with m_c1:
                                    st.metric("Latency", f"{trace.latency_ms:.0f} ms", delta=f"{active_tc.latency_budget - trace.latency_ms:.0f} ms within budget" if trace.latency_ms <= active_tc.latency_budget else "Exceeded budget", delta_color="normal" if trace.latency_ms <= active_tc.latency_budget else "inverse")
                                with m_c2:
                                    st.metric("Tools Called", ", ".join(trace.tools_called) if trace.tools_called else "None")
                                with m_c3:
                                    st.metric("Weighted Score", f"{c_summary.get('weighted_score', 0):.1f}%")
                                with m_c4:
                                    st.metric("Overall Verdict", "PASSED" if c_summary.get("passed") else "FAILED")

                                st.markdown("##### Detailed Metric Breakdown:")
                                for r in eval_results:
                                    icon = "✅" if r.passed else "❌"
                                    e_type = str(getattr(r, "evaluation_type", "deterministic")).lower()
                                    if getattr(r, "is_mock", False) or trace.is_mock:
                                        t_badge = "[MOCK]"
                                        t_color = "#f59e0b"
                                    elif e_type in ["llm_based", "llm_judge", "model_based"]:
                                        t_badge = "[LLM-BASED]"
                                        t_color = "#10b981"
                                    elif e_type in ["heuristic", "semantic"]:
                                        t_badge = "[HEURISTIC]"
                                        t_color = "#c084fc"
                                    else:
                                        t_badge = "[DETERMINISTIC]"
                                        t_color = "#38bdf8"
                                    st.markdown(
                                        f"- {icon} **{r.metric}** <span style='font-size:10px; font-weight:700; color:{t_color};'>{t_badge}</span> — Score: `{r.score:.2f}` — {r.explanation or r.details}",
                                        unsafe_allow_html=True,
                                    )

                                # Render Failure Analysis if test failed or has failing checks
                                render_failure_analysis(active_tc, trace, eval_results, show_all=True)

                            except Exception as ex:
                                st.error(f"Execution failed: {str(ex)}")
                else:
                    st.warning("No active agents found in registry.")

            with tab_edit:
                with st.form(f"edit_form_{active_tc.test_id}"):
                    ed_c1, ed_c2 = st.columns(2)
                    with ed_c1:
                        ed_name = st.text_input("Name*", value=active_tc.name)
                        ed_query = st.text_area("User Input (Query)*", value=active_tc.user_input, height=85)
                        ed_behavior = st.text_area("Expected Behavior*", value=active_tc.expected_behavior, height=85)
                    with ed_c2:
                        ed_tool = st.text_input("Expected Tool (Optional)", value=active_tc.expected_tool or "")
                        cur_kws = ", ".join(active_tc.expected_keywords) if active_tc.expected_keywords else (active_tc.expected_answer or "")
                        ed_kws = st.text_input("Expected Answer / Keywords (Optional)", value=cur_kws)
                        ed_lat = st.number_input("Latency Budget (ms)*", min_value=100.0, max_value=30000.0, value=float(active_tc.latency_budget), step=500.0)
                        ed_en = st.checkbox("Enabled (Active in test suite)", value=active_tc.enabled)

                    if st.form_submit_button("Save Test Case Changes", type="primary", width="stretch"):
                        try:
                            parsed_kws = [k.strip() for k in ed_kws.split(",") if k.strip()] if ed_kws.strip() else []
                            tc_manager.update_test_case(active_tc.test_id, {
                                "name": ed_name.strip(),
                                "user_input": ed_query.strip(),
                                "expected_behavior": ed_behavior.strip(),
                                "expected_tool": ed_tool.strip() if ed_tool.strip() else None,
                                "expected_keywords": parsed_kws,
                                "expected_answer": ed_kws.strip() if ed_kws.strip() else None,
                                "latency_budget": float(ed_lat),
                                "enabled": ed_en,
                            })
                            st.success(f"✅ Successfully updated test case '{active_tc.test_id}'!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update test case: {str(e)}")
