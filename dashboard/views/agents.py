"""
Section 2: Agents - Agent Registry, Version Management, and Secure Sandbox Upload/Execution.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.registry.registry import AgentRegistry
from src.sandbox import (
    SandboxManager,
    SandboxConfig,
    SecurityLevel,
    SecurityRiskLevel,
    SecurityCategory,
    ArchiveValidator,
    ArchiveValidationError,
    StaticInspectionResult,
)
from src.security import (
    SecurityInputValidator,
    SecurityValidationError,
)


from dashboard.views.common import render_section_header, render_kpi_card, navigate_to


def render_agents(registry: AgentRegistry, sandbox_manager: SandboxManager):
    render_section_header(
        title="Agent Registry & Secure Sandbox",
        subtitle="Manage versions, configurations, and adapter protocols or audit arbitrary agent projects in an isolated sandbox.",
        breadcrumb="MANAGEMENT // AGENTS",
        action_badge="ISOLATED BOUNDARY",
    )

    # Quick Onboarding Shortcut
    q_c1, q_c2 = st.columns([4.2, 1.3])
    with q_c1:
        st.markdown(
            """ <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(168, 85, 247, 0.1) 100%); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 16px; margin-bottom: 14px;"> <span style="font-size: 13px; font-weight: 700; color: #f8fafc;">⚡ Connecting an external agent via SDK?</span> <span style="font-size: 12px; color: #cbd5e1; margin-left: 6px;">Use the streamlined onboarding workflow to configure API keys, generate code snippets, and verify telemetry.</span> </div> """,
            unsafe_allow_html=True,
        )
    with q_c2:
        if st.button("⚡ Connect Agent ➔", key="btn_quick_connect_agent_banner", width="stretch", type="primary"):
            navigate_to("⚡ Connect Agent")

    agents_tab1, agents_tab2, agents_tab3 = st.tabs([
        "📋 Registered Agents & Versions",
        "➕ Register New Agent",
        "🛡️ Secure Sandbox & Project Upload",
    ])

    agents = registry.list_agents()

    # ---------------------------------------------------------------------------
    # TAB 1: Registered Agents & Versions
    # ---------------------------------------------------------------------------
    with agents_tab1:
        # Top KPI summary
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            render_kpi_card("Registered Agents", str(len(agents)), subtitle="Configured Adapters", icon="🤖", accent_color="#38bdf8")
        with kpi_c2:
            render_kpi_card("Active Agents", str(sum(1 for a in agents if a.status == "active")), subtitle="Available for Benchmarks", icon="🟢", accent_color="#10b981")
        with kpi_c3:
            render_kpi_card("Disabled Agents", str(sum(1 for a in agents if a.status == "disabled")), subtitle="Offline", icon="⚪", accent_color="#64748b")

        st.divider()

        st.markdown(
            """ <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 12px;"> <span style="font-size: 15px; font-weight: 700; color: #f8fafc; letter-spacing: 0.02em;">Registered Agents Catalog</span> <span style="font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Total: %d Adapters</span> </div> """ % len(agents),
            unsafe_allow_html=True,
        )

        if agents:
            for a in agents:
                is_active = (a.status == "active")
                status_color = "#10b981" if is_active else "#64748b"
                status_text = "ACTIVE" if is_active else "DISABLED"
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3.5, 2.5, 2.5, 2])
                    with col1:
                        st.markdown(
                            f""" <div style="display: flex; align-items: center; gap: 10px;"> <div style="width: 36px; height: 36px; border-radius: 9px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); display: flex; align-items: center; justify-content: center; font-size: 18px;"> 🤖 </div> <div> <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">{a.name}</div> <div style="font-size: 11px; color: #64748b; font-family: 'JetBrains Mono', monospace;">ID: {a.agent_id[:16]}...</div> </div> </div> """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f""" <div style="font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px;">FRAMEWORK & TYPE</div> <div> <span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 2px 7px; border-radius: 5px; font-size: 11px; font-weight: 600;">{a.framework or 'custom'}</span> <span style="background: rgba(148, 163, 184, 0.1); color: #94a3b8; padding: 2px 6px; border-radius: 5px; font-size: 10.5px;">{a.integration_type}</span> </div> """,
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.markdown(
                            f""" <div style="font-size: 10.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px;">MODEL & VERSION</div> <div> <span style="color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;">{a.provider_model or 'default'}</span> <span style="color: #64748b; font-size: 11px; font-weight: 600;">({a.active_version})</span> </div> """,
                            unsafe_allow_html=True,
                        )
                    with col4:
                        st.markdown(
                            f""" <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;"> <span style="display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 9999px; background: {status_color}18; border: 1px solid {status_color}40; color: {status_color}; font-size: 11px; font-weight: 700; letter-spacing: 0.04em;"> <span style="width: 6px; height: 6px; border-radius: 50%; background: {status_color}; box-shadow: 0 0 6px {status_color};"></span>{status_text} </span> </div> """,
                            unsafe_allow_html=True,
                        )

            st.markdown("<div style='margin-top: 20px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

            # Detailed Agent & Version Management
            st.subheader("🔍 Manage Agent Details & Versions")
            selected_agent_name = st.selectbox(
                "Select Agent to Manage",
                [a.name for a in agents],
                index=0,
                key="manage_agent_selector"
            )
            agent_obj = registry.get_agent(selected_agent_name)

            if agent_obj:
                with st.container(border=True):
                    dt_c1, dt_c2 = st.columns(2)
                    with dt_c1:
                        st.markdown(f"**Agent ID:** `{agent_obj.agent_id}`")
                        st.markdown(f"**Name:** `{agent_obj.name}`")
                        st.markdown(f"**Description:** {agent_obj.description or 'No description provided.'}")
                        st.markdown(f"**Status:** {'🟢 Active' if agent_obj.status == 'active' else '🔴 Disabled'}")
                    with dt_c2:
                        st.markdown(f"**Integration Type:** `{agent_obj.integration_type}`")
                        st.markdown(f"**Framework:** `{agent_obj.framework}`")
                        st.markdown(f"**Active Version:** `{agent_obj.active_version}`")

                        # Enable / Disable Toggle
                        if agent_obj.status == "active":
                            if st.button("🔴 Disable Agent", key="btn_disable_agent"):
                                registry.set_agent_status(agent_obj.agent_id, "disabled")
                                st.rerun()
                        else:
                            if st.button("🟢 Enable Agent", key="btn_enable_agent"):
                                registry.set_agent_status(agent_obj.agent_id, "active")
                                st.rerun()

                st.markdown("#### Version History")
                ver_records = registry.get_agent_versions(agent_obj.agent_id)
                for v in ver_records:
                    is_active = (v.version == agent_obj.active_version)
                    v_accent = "#38bdf8" if is_active else "#64748b"
                    with st.container(border=True):
                        vc1, vc2, vc3 = st.columns([2, 3, 2])
                        with vc1:
                            st.markdown(f"**Version:** `{v.version}` {'⭐ (ACTIVE)' if is_active else ''}")
                            st.caption(f"Created: {v.created_at}")
                        with vc2:
                            st.markdown(f"**Model:** `{v.model}`")
                            st.caption(f"Config: `{v.configuration_metadata}`")
                        with vc3:
                            if not is_active:
                                if st.button(f"Set Active ({v.version})", key=f"btn_activate_{v.version}"):
                                    registry.set_active_version(agent_obj.agent_id, v.version)
                                    st.rerun()

                # Change active version
                all_ver_tags = [v.version for v in ver_records]
                set_c1, set_c2 = st.columns([3, 1])
                with set_c1:
                    default_idx = all_ver_tags.index(agent_obj.active_version) if agent_obj.active_version in all_ver_tags else 0
                    new_active_v = st.selectbox("Set Active Version", all_ver_tags, index=default_idx, key="active_ver_sel")
                with set_c2:
                    if st.button("Apply Version", key="btn_set_ver_action"):
                        registry.set_active_version(agent_obj.agent_id, new_active_v)
                        st.success(f"Active version set to `{new_active_v}`!")
                        st.rerun()

                # Add New Version to this agent
                with st.expander(f"➕ Add New Version to '{agent_obj.name}'"):
                    with st.form("add_version_form_agent"):
                        v_tag = st.text_input("Version Tag*", placeholder="e.g. v2.0")
                        v_model = st.text_input("Model Override", value=agent_obj.provider_model)
                        v_config_raw = st.text_area("Configuration JSON (optional)", value="{}")
                        v_active_check = st.checkbox("Set as active version immediately", value=True)

                        if st.form_submit_button("Add Version"):
                            try:
                                parsed_cfg = json.loads(v_config_raw) if v_config_raw.strip() else {}
                                registry.add_version(
                                    agent_id=agent_obj.agent_id,
                                    version=v_tag.strip(),
                                    model=v_model.strip(),
                                    config=parsed_cfg,
                                    set_active=v_active_check,
                                )
                                st.success(f"Added version `{v_tag}` to '{agent_obj.name}'!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to add version: {str(e)}")
        else:
            st.info("No registered agents found.")

    # ---------------------------------------------------------------------------
    # TAB 2: Register New Agent
    # ---------------------------------------------------------------------------
    with agents_tab2:
        st.subheader("Register a New AI Agent")
        st.caption("Add an agent adapter (mock, local python callable, or HTTP endpoint) to the benchmark registry.")

        with st.form("add_agent_form_tab2"):
            form_c1, form_c2 = st.columns(2)
            with form_c1:
                new_name = st.text_input("Agent Name*", placeholder="e.g. Support RAG Agent")
                new_desc = st.text_area("Description", placeholder="Answers inquiries using vector search & ticketing tools")
                new_framework = st.selectbox("Framework", ["langgraph", "crewai", "autogen", "custom", "http_api", "langchain"])
            with form_c2:
                new_itype = st.selectbox(
                    "Integration Type*",
                    ["mock", "local_python", "http_api", "sdk"],
                    format_func=lambda x: {
                        "mock": "Mock / Internal Agent (Deterministic / Zero API cost)",
                        "local_python": "Local Python Callable (Module path or function)",
                        "http_api": "HTTP / REST API Agent (POST Endpoint)",
                        "sdk": "AgentPulse SDK (Direct telemetry ingestion via @observe)",
                    }[x]
                )
                new_model = st.text_input("Provider / Model", value="claude-3-5-haiku")
                new_version = st.text_input("Initial Version Tag", value="v1.0")

            st.markdown("**Adapter Configuration Parameters:**")
            config_dict = {}
            if new_itype == "mock":
                mock_bug = st.checkbox("Simulate injected tool routing bug (for regression testing demo)")
                config_dict["inject_bug"] = mock_bug
            elif new_itype == "local_python":
                py_c1, py_c2 = st.columns(2)
                with py_c1:
                    py_mod = st.text_input("Module Path*", value="src.agent.custom_agent_template", help="Python import path to the module")
                with py_c2:
                    py_fn = st.text_input("Callable Function Name*", value="run_custom_agent", help="Function that takes (task_id, query) and returns output or Trace")
                config_dict["module_path"] = py_mod
                config_dict["callable_name"] = py_fn
            elif new_itype == "http_api":
                http_url = st.text_input("Endpoint URL*", placeholder="https://api.mycompany.com/v1/agent")
                http_c1, http_c2 = st.columns(2)
                with http_c1:
                    http_timeout = st.number_input("Timeout (seconds)", value=30.0, min_value=1.0, max_value=300.0)
                with http_c2:
                    http_token = st.text_input("Bearer Token (optional)", type="password")
                config_dict["endpoint_url"] = http_url
                config_dict["timeout"] = http_timeout
                if http_token:
                    config_dict["headers"] = {"Authorization": f"Bearer {http_token}"}
            elif new_itype == "sdk":
                st.info("📡 **AgentPulse SDK Agent**: Telemetry is automatically streamed into AgentPulse whenever your agent executes with `@agentpulse.observe`. Ingestion endpoint is active at `/v1/spans`.")
                config_dict["integration_type"] = "sdk"

            submitted = st.form_submit_button("Register Agent", width="stretch")
            if submitted:
                if not new_name.strip():
                    st.error("Agent Name is required.")
                else:
                    try:
                        agent_rec = registry.register_agent(
                            name=new_name.strip(),
                            description=new_desc.strip(),
                            framework=new_framework,
                            provider_model=new_model.strip(),
                            integration_type=new_itype,
                            initial_version=new_version.strip() or "v1.0",
                            config=config_dict,
                        )
                        st.success(f"✅ Successfully registered agent '{agent_rec.name}' (ID: `{agent_rec.agent_id}`)!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Registration failed: {str(ex)}")

    # ---------------------------------------------------------------------------
    # TAB 3: Secure Sandbox & Project Upload
    # ---------------------------------------------------------------------------
    with agents_tab3:
        st.subheader("🛡️ Secure Sandbox Execution & Static Code Inspector")
        st.caption("Safely unpack, inspect, and evaluate arbitrary AI agent repositories in an isolated process.")

        # Architecture Overview Cards
        sb_c1, sb_c2, sb_c3, sb_c4 = st.columns(4)
        sb_c1.info("**1. Ingestion & Validation**\n\nZipSlip path traversal checks, zip-bomb size bounds (25MB cap), binary blocks.")
        sb_c2.info("**2. Static AST Analysis**\n\nDetects `eval()`, shell spawns, dangerous imports, and sensitive env keys.")
        sb_c3.info("**3. Isolation Boundary**\n\nEphemeral subprocess with sanitized host env (zero host secrets exposed).")
        sb_c4.info("**4. Destruction**\n\nStrict execution timeout, 500KB output cap, and guaranteed cleanup.")

        with st.expander("🛡️ Security Safeguards & Disclosures", expanded=False):
            st.markdown(
                """ - **Secret Redaction Active**: Heuristic scrubbers purge OpenAI, Anthropic, Gemini, AWS, GitHub, DB URIs, and private keys. - **Upload Guardrails**: Max compressed size: **25 MB**, max uncompressed: **50 MB**, max files: **500**. Blocked executable binaries. - **Process Isolation**: Subprocess execution with purged environment prevents untrusted code from stealing host credentials. - **Production Notice**: Operates on host OS subprocess isolation. For untrusted public multi-tenant code, run inside rootless Docker containers or gVisor microVMs. """
            )

        st.divider()

        # Ingestion Source
        st.subheader("📂 1. Ingest Agent Project Package")
        upload_choice = st.segmented_control(
            "Ingestion Source",
            ["Upload ZIP Archive", "Existing Workspace Directory", "📦 Load Built-in Demo Agent Sample"],
            default="Upload ZIP Archive",
            key="sandbox_upload_method"
        )
        upload_method = upload_choice if upload_choice is not None else "Upload ZIP Archive"

        staged_project_dir = None
        project_label = ""

        if upload_method == "Upload ZIP Archive":
            uploaded_file = st.file_uploader("Upload Agent Package (.zip)", type=["zip"], help="Upload an agent codebase ZIP archive (Max 25 MB).", key="agent_zip_uploader")
            if uploaded_file is not None:
                try:
                    SecurityInputValidator.validate_upload_archive_size(uploaded_file)
                    clean_name = SecurityInputValidator.sanitize_filename(uploaded_file.name)
                    temp_zip_path = os.path.join(tempfile.gettempdir(), f"upload_{clean_name}")
                    with open(temp_zip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    extract_target = os.path.join(tempfile.gettempdir(), "agent_eval_sandboxes", f"uploaded_{int(time.time())}")
                    validator = ArchiveValidator()
                    staged_project_dir = validator.validate_and_extract(temp_zip_path, extract_target)
                    project_label = clean_name
                    st.success(f"✅ Archive validated ({len(uploaded_file.getbuffer()) / 1024:.1f} KB) and unpacked into `{staged_project_dir}`")
                except (ArchiveValidationError, SecurityValidationError) as ave:
                    st.error(f"❌ Security Validation Violation: {str(ave)}")
                except Exception as e:
                    st.error(f"❌ Failed to extract archive: {str(e)}")
        elif upload_method == "📦 Load Built-in Demo Agent Sample":
            sample_dir = Path("./src/agent").resolve()
            if sample_dir.is_dir():
                staged_project_dir = sample_dir
                project_label = "Demo ReAct Agent (src/agent)"
                st.info(f"Loaded built-in demo agent workspace from `{sample_dir}`")
        else:
            dir_input = st.text_input(
                "Project Directory Path",
                value="src/agent",
                placeholder="e.g. ./src/agent",
                key="sandbox_dir_input"
            )
            if dir_input and os.path.isdir(dir_input):
                try:
                    validator = ArchiveValidator()
                    validator.validate_directory(dir_input)
                    staged_project_dir = Path(dir_input).resolve()
                    project_label = os.path.basename(staged_project_dir)
                    st.success(f"✅ Validated directory structure for `{staged_project_dir}`")
                except ArchiveValidationError as ave:
                    st.error(f"❌ Security Validation Violation: {str(ave)}")
                except Exception as e:
                    st.error(f"❌ Directory validation error: {str(e)}")

        if staged_project_dir:
            st.divider()
            st.subheader(f"🔍 2. Static AST Inspection & Security Audit: {project_label}")

            with st.spinner("Analyzing syntax trees, imports, frameworks, and security boundaries..."):
                inspection: StaticInspectionResult = sandbox_manager.inspect_project(Path(staged_project_dir))

            manifest = inspection.manifest

            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            risk_color = "🟢" if inspection.overall_risk in [SecurityRiskLevel.SAFE, SecurityRiskLevel.LOW] else ("🟡" if inspection.overall_risk == SecurityRiskLevel.MEDIUM else "🔴")
            r_col1.metric("Security Risk Level", f"{risk_color} {inspection.overall_risk.value}")
            r_col2.metric("Detected Framework", manifest.framework.upper())
            r_col3.metric("Files / LOC", f"{inspection.file_count} files / {inspection.lines_of_code} LOC")
            r_col4.metric("Security Alerts", len(inspection.alerts))

            if inspection.overall_risk == SecurityRiskLevel.CRITICAL:
                st.error(
                    f"🚨 **CRITICAL SECURITY RISK DETECTED**\n\n{inspection.rejection_reason}\n\n"
                    "This project contains hazardous operations (e.g. `eval()`, `exec()`, raw shell execution). Execution restricted."
                )
            elif inspection.overall_risk == SecurityRiskLevel.HIGH:
                st.warning(f"⚠️ **HIGH SECURITY RISK DETECTED**\n\n{inspection.rejection_reason}\n\nProceed with caution.")
            else:
                st.success("✅ **STATIC AUDIT PASSED**: No critical security violations detected. Safe to execute in isolated sandbox.")

            m_tab1, m_tab2, m_tab3 = st.tabs(["📋 Manifest & Entry Point", "🚨 Security Alerts & AST Findings", "📦 Dependencies & Env Vars"])

            with m_tab1:
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    py_files_opts = manifest.python_files if manifest.python_files else [manifest.entry_point]
                    default_file_idx = py_files_opts.index(manifest.entry_point) if manifest.entry_point in py_files_opts else 0
                    selected_entry_point = st.selectbox("Entry Point File", py_files_opts, index=default_file_idx, key="sb_entry_file")
                    selected_entry_symbol = st.text_input("Entry Function or Class", value=manifest.entry_symbol, key="sb_entry_symbol")
                with e_col2:
                    st.markdown(f"**Framework:** `{manifest.framework}`")
                    st.markdown(f"**Description:** {manifest.description}")
                    st.markdown(f"**Workspace Location:** `{staged_project_dir}`")

            with m_tab2:
                if inspection.alerts:
                    alert_rows = []
                    for a in inspection.alerts:
                        alert_rows.append({
                            "Rule ID": a.rule_id,
                            "Risk Level": a.risk_level.value,
                            "Category": a.category.value,
                            "File": a.file_path,
                            "Line": a.line_number or "N/A",
                            "Message": a.message,
                        })
                    st.dataframe(pd.DataFrame(alert_rows), width="stretch")
                else:
                    st.info("No security alerts raised during AST inspection.")

            with m_tab3:
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown("##### Detected Dependencies")
                    if manifest.dependencies:
                        for dep in manifest.dependencies:
                            st.markdown(f"- `{dep}`")
                    else:
                        st.caption("No explicit dependencies found in requirements.txt.")
                with d_col2:
                    st.markdown("##### Required Environment Variables")
                    if manifest.required_env_vars:
                        for ev in manifest.required_env_vars:
                            st.markdown(f"- 🔑 `{ev}`")
                    else:
                        st.caption("No dynamic environment variable accesses detected.")

            st.divider()

            # Safeguards & Isolation Configuration
            st.subheader("⚙️ 3. Sandbox Safeguards & Isolation Configuration")
            cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4)
            with cfg_c1:
                timeout_sec = st.slider("Timeout Limit (seconds)", min_value=1.0, max_value=120.0, value=15.0, step=1.0, key="sb_timeout")
            with cfg_c2:
                mem_limit_mb = st.selectbox("Memory Limit (MB)", [256, 512, 1024, 2048], index=1, key="sb_mem")
            with cfg_c3:
                net_policy = st.selectbox("Network Policy", ["ALLOW_ALL", "RESTRICTED", "OFFLINE"], index=0, key="sb_net")
            with cfg_c4:
                max_out_kb = st.number_input("Max Output Buffer (KB)", min_value=10, max_value=5000, value=500, step=50, key="sb_out_kb")

            with st.expander("🔑 Inject Custom Environment Variables (Host Secrets are ALWAYS Scrubbed)", expanded=False):
                custom_env_json = st.text_area("Environment JSON", value='{\n  "MOCK_MODE": "true"\n}', height=100, key="sb_env_json")

            parsed_custom_env = {}
            if custom_env_json.strip():
                try:
                    parsed_custom_env = json.loads(custom_env_json)
                except Exception:
                    st.error("Invalid JSON for custom environment variables.")

            sandbox_config = SandboxConfig(
                timeout_seconds=float(timeout_sec),
                max_memory_mb=int(mem_limit_mb),
                max_output_size_bytes=int(max_out_kb * 1024),
                custom_env_vars=parsed_custom_env,
            )

            st.divider()

            # Interactive Sandbox Execution
            st.subheader("⚡ 4. Test Run in Isolated Sandbox")
            test_prompt = st.text_area(
                "Test Query / Task Input",
                value="What is the stock price of Apple, and calculate a 15% discount on $200?",
                height=80,
                key="sb_test_prompt"
            )

            if test_prompt:
                try:
                    SecurityInputValidator.validate_text_input(test_prompt, "Test Query")
                    inj_check = SecurityInputValidator.scan_prompt_injection(test_prompt)
                    if inj_check.is_suspicious:
                        st.warning(f"⚠️ **Security Notice**: Detected prompt pattern: `{', '.join(inj_check.detected_patterns)}` (Risk score: {inj_check.risk_score:.2f}). Workload will execute inside the isolated sandbox subprocess.")
                except SecurityValidationError as sve:
                    st.error(f"Input Validation Error: {str(sve)}")

            run_sandbox_btn = st.button("🚀 Execute in Sandbox", type="primary", width="stretch", key="btn_run_sandbox")

            if run_sandbox_btn:
                with st.spinner("Provisioning ephemeral workspace, launching isolated child process, and capturing telemetry..."):
                    exec_result = sandbox_manager.execute_task(
                        project_dir=Path(staged_project_dir),
                        task_id=f"test_run_{int(time.time())}",
                        query=test_prompt,
                        entry_point=selected_entry_point.strip(),
                        entry_symbol=selected_entry_symbol.strip(),
                        config=sandbox_config,
                    )

                st.markdown("#### Execution Results")
                out_c1, out_c2, out_c3, out_c4 = st.columns(4)
                stat_ic = "✅" if exec_result.success else "❌"
                out_c1.metric("Status", f"{stat_ic} {exec_result.status}")
                out_c2.metric("Latency", f"{exec_result.duration_ms:.1f} ms")
                out_c3.metric("Isolation Level", exec_result.security_level.value)
                out_c4.metric("Tokens (In / Out)", f"{exec_result.token_usage.get('input_tokens', 0)} / {exec_result.token_usage.get('output_tokens', 0)}")

                if exec_result.timed_out:
                    st.error(f"⏱️ **Execution Timed Out**: Workload exceeded limit of {sandbox_config.timeout_seconds}s and was forcefully terminated.")

                if exec_result.error:
                    st.error(f"**Error Details:**\n\n```\n{exec_result.error}\n```")

                st.markdown("##### Agent Output Response:")
                st.info(exec_result.response if exec_result.response else "*(No response text returned)*")

                if exec_result.trace_steps:
                    st.markdown("##### Captured Telemetry Steps:")
                    for step in exec_result.trace_steps:
                        st.markdown(f"- **Step {step.get('step_index', 1)} ({step.get('step_type', 'agent')})**: `{step.get('tool_name', 'N/A')}` — *{step.get('latency_ms', 0):.1f}ms*")

                with st.expander("View Raw Subprocess Stdout / Stderr"):
                    st.code(
                        f"STDOUT:\n{exec_result.raw_output.get('stdout', '')}\n\nSTDERR:\n{exec_result.raw_output.get('stderr', '')}",
                        language="text",
                    )

            st.divider()

            # Register Sandboxed Agent
            st.subheader("🤖 5. Register Sandboxed Agent into Registry")
            with st.form("register_sandboxed_agent_form"):
                reg_c1, reg_c2, reg_c3 = st.columns(3)
                with reg_c1:
                    reg_agent_id = st.text_input("Agent ID", value=f"sandboxed_{project_label.lower().replace('.zip', '').replace(' ', '_')}")
                with reg_c2:
                    reg_agent_name = st.text_input("Display Name", value=f"Sandboxed: {project_label}")
                with reg_c3:
                    reg_version = st.text_input("Version Tag", value="v1.0")

                if st.form_submit_button("Register to Agent Registry", type="primary", width="stretch"):
                    try:
                        registered = registry.register_agent(
                            name=reg_agent_name.strip(),
                            description=f"Sandboxed agent ({manifest.framework}) from '{project_label}' with isolated execution boundary.",
                            framework=manifest.framework,
                            integration_type="sandboxed",
                            initial_version=reg_version.strip(),
                            config={
                                "type": "sandboxed",
                                "project_dir": str(staged_project_dir),
                                "entry_point": selected_entry_point.strip(),
                                "entry_symbol": selected_entry_symbol.strip(),
                                "timeout_seconds": float(timeout_sec),
                                "framework": manifest.framework,
                            },
                        )
                        st.success(f"🎉 Successfully registered **{registered.name}** (`{registered.agent_id}`) into the Agent Registry!")
                    except Exception as ex:
                        st.error(f"Failed to register agent: {str(ex)}")
