"""
Connect Agent Onboarding View.

Provides a fast, hackathon-friendly workflow for connecting external agents:
  1. Agent Name & Version registration
  2. API Key creation
  3. SDK installation instructions
  4. Integration example code snippet
  5. Live Connection Status & "Test Connection" verification action
"""

import json
import os
import secrets
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import streamlit as st

from src.storage.db import init_db, get_session
from src.storage.models import AgentRecord, AgentVersionRecord
from dashboard.views.common import render_section_header, navigate_to


def _check_backend_health(url: str, timeout: float = 1.5) -> Dict[str, Any]:
    """Check if the AgentPulse Ingestion API is responding."""
    health_url = f"{url.rstrip('/')}/health"
    start = time.time()
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "AgentPulse-Console/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.time() - start) * 1000
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {"online": True, "latency_ms": elapsed_ms, "data": data, "error": None}
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {"online": False, "latency_ms": elapsed_ms, "data": None, "error": str(e)}
    return {"online": False, "latency_ms": 0, "data": None, "error": "Unknown status"}


def _send_test_span(backend_url: str, api_key: str, agent_id: str, agent_version: str) -> Dict[str, Any]:
    """Send a test telemetry span to verify the end-to-end ingestion pipeline."""
    spans_url = f"{backend_url.rstrip('/')}/v1/spans"
    trace_id = f"trc_test_{secrets.token_hex(4)}"
    span_id = f"spn_test_{secrets.token_hex(4)}"
    run_id = f"run_test_{secrets.token_hex(4)}"

    payload = {
        "trace_id": trace_id,
        "span_id": span_id,
        "function_name": "test_connection_ping",
        "input": "AgentPulse Connection Test Ping",
        "output": "Connection verified successfully! Telemetry pipeline active.",
        "start_time": time.time() - 0.02,
        "end_time": time.time(),
        "latency": 20.0,
        "status": "success",
        "span_type": "agent",
        "agent_id": agent_id,
        "agent_name": agent_id,
        "agent_version": agent_version,
        "run_id": run_id,
        "metadata": {
            "source": "connect_agent_ui_test",
            "environment": "demo",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AgentPulse-TestConnection/1.0",
    }
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(spans_url, data=data_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            elapsed_ms = (time.time() - start) * 1000
            resp_body = json.loads(resp.read().decode("utf-8"))
            return {
                "success": True,
                "status_code": resp.status,
                "latency_ms": elapsed_ms,
                "response": resp_body,
                "trace_id": trace_id,
                "run_id": resp_body.get("run_id") or run_id,
                "error": None,
            }
    except urllib.error.HTTPError as he:
        elapsed_ms = (time.time() - start) * 1000
        err_body = he.read().decode("utf-8") if he.fp else str(he)
        return {
            "success": False,
            "status_code": he.code,
            "latency_ms": elapsed_ms,
            "response": None,
            "error": f"HTTP {he.code}: {err_body}",
        }
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "success": False,
            "status_code": 0,
            "latency_ms": elapsed_ms,
            "response": None,
            "error": str(e),
        }


def _register_agent_in_db(agent_name: str, agent_version: str, api_key: str) -> str:
    """Ensure the agent and its version exist in SQLite storage."""
    init_db()
    session = get_session()
    agent_id = agent_name.strip().lower().replace(" ", "-").replace("_", "-")
    try:
        existing = session.query(AgentRecord).filter_by(agent_id=agent_id).first()
        if not existing:
            agent_rec = AgentRecord(
                agent_id=agent_id,
                name=agent_name.strip(),
                description="External agent connected via AgentPulse SDK onboarding workflow",
                framework="sdk",
                provider_model="claude-3-5-haiku",
                integration_type="sdk",
                status="active",
                active_version=agent_version.strip() or "1.0",
            )
            session.add(agent_rec)
        else:
            existing.integration_type = "sdk"
            existing.active_version = agent_version.strip() or existing.active_version
            existing.status = "active"

        # Register version
        ver_tag = agent_version.strip() or "1.0"
        ver_id = f"ver_{agent_id}_{ver_tag.replace('.', '_')}"
        existing_ver = session.query(AgentVersionRecord).filter_by(version_id=ver_id).first()
        if not existing_ver:
            ver_rec = AgentVersionRecord(
                version_id=ver_id,
                agent_id=agent_id,
                version=ver_tag,
                model="claude-3-5-haiku",
                configuration_metadata=json.dumps({"integration": "sdk", "api_key_configured": bool(api_key)}),
                status="active",
            )
            session.add(ver_rec)

        session.commit()
        return agent_id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def render_connect_agent():
    """Render the Connect Agent onboarding experience."""
    render_section_header(
        title="Connect Agent",
        subtitle="Quickstart onboarding: register your external agent, generate SDK integration code, and verify live connection in seconds.",
        breadcrumb="CORE PLATFORM // CONNECT AGENT",
        action_badge="HACKATHON READY",
    )

    # State initialization
    if "ca_agent_name" not in st.session_state:
        st.session_state["ca_agent_name"] = "customer-support"
    if "ca_agent_version" not in st.session_state:
        st.session_state["ca_agent_version"] = "1.0"
    if "ca_api_key" not in st.session_state:
        # Default to environment key or generate a clean demo token
        env_key = os.getenv("AGENTPULSE_API_KEY", "").strip()
        st.session_state["ca_api_key"] = env_key or f"ap_live_{secrets.token_hex(8)}"
    if "ca_backend_url" not in st.session_state:
        st.session_state["ca_backend_url"] = os.getenv("AGENTPULSE_BACKEND_URL", "http://localhost:8000").rstrip("/")
    if "ca_registered_id" not in st.session_state:
        st.session_state["ca_registered_id"] = "customer-support"
        # Pre-register default on first view
        try:
            _register_agent_in_db("customer-support", "1.0", st.session_state["ca_api_key"])
        except Exception:
            pass

    # Check live backend health
    backend_status = _check_backend_health(st.session_state["ca_backend_url"])

    # ---------------------------------------------------------------------------
    # Step 1: Configuration & Registration Form
    # ---------------------------------------------------------------------------
    st.markdown(
        """ <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 18px 22px; margin-bottom: 20px;"> <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;"> <div style="display: flex; align-items: center; gap: 10px;"> <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(56, 189, 248, 0.15); display: flex; align-items: center; justify-content: center; font-size: 16px;"> ⚡ </div> <div> <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">Register Agent Credentials</div> <div style="font-size: 11.5px; color: #94a3b8;">Enter your agent's identity to generate tailored SDK integration code.</div> </div> </div> </div> """,
        unsafe_allow_html=True,
    )

    with st.form("connect_agent_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            in_name = st.text_input("Agent Name*", value=st.session_state["ca_agent_name"], help="Name or slug for your external AI agent.")
            in_version = st.text_input("Agent Version*", value=st.session_state["ca_agent_version"], help="Version string for prompt/model tracking.")
        with fc2:
            in_key = st.text_input("API Key*", value=st.session_state["ca_api_key"], help="Telemetry API key sent in X-API-Key or Authorization header.")
            in_url = st.text_input("Backend URL*", value=st.session_state["ca_backend_url"], help="AgentPulse HTTP Ingestion Server endpoint.")

        submitted = st.form_submit_button("⚡ Register & Update Snippet", type="primary", width="stretch")
        if submitted:
            st.session_state["ca_agent_name"] = in_name.strip()
            st.session_state["ca_agent_version"] = in_version.strip()
            st.session_state["ca_api_key"] = in_key.strip()
            st.session_state["ca_backend_url"] = in_url.strip().rstrip("/")
            try:
                reg_id = _register_agent_in_db(in_name, in_version, in_key)
                st.session_state["ca_registered_id"] = reg_id
                st.toast(f"✅ Agent '{reg_id}' registered successfully in registry!", icon="🚀")
            except Exception as e:
                st.error(f"Error registering agent: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    agent_id = st.session_state.get("ca_registered_id", "customer-support")
    agent_name = st.session_state.get("ca_agent_name", "customer-support")
    agent_version = st.session_state.get("ca_agent_version", "1.0")
    api_key = st.session_state.get("ca_api_key", "ap_live_key")
    backend_url = st.session_state.get("ca_backend_url", "http://localhost:8000")

    # ---------------------------------------------------------------------------
    # Step 2: 5 Required Status & Onboarding Elements
    # ---------------------------------------------------------------------------
    c_status1, c_status2, c_status3 = st.columns([1.5, 1.5, 2])

    # Element 1: Agent Registered
    with c_status1:
        st.markdown(
            f""" <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px 16px; min-height: 110px;"> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"> <span style="font-size: 11px; font-weight: 700; color: #34d399; text-transform: uppercase;">1. Agent Registered</span> <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">ACTIVE</span> </div> <div style="font-size: 15px; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">{agent_name}</div> <div style="font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;"> ID: <span style="color: #38bdf8;">{agent_id}</span> · v{agent_version} </div> <div style="margin-top: 4px; font-size: 10.5px; color: #c084fc; font-weight: 600;"> Integration: SDK (@observe) </div> </div> """,
            unsafe_allow_html=True,
        )

    # Element 2: API Key Created
    with c_status2:
        masked_key = (api_key[:8] + "••••••••" + api_key[-4:]) if len(api_key) > 12 else (api_key[:4] + "••••")
        st.markdown(
            f""" <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 10px; padding: 14px 16px; min-height: 110px;"> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"> <span style="font-size: 11px; font-weight: 700; color: #c084fc; text-transform: uppercase;">2. API Key Created</span> <span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">SECURE</span> </div> <div style="font-size: 13px; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;"> {masked_key} </div> <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;"> Header: <code style="color: #38bdf8; font-size: 10.5px;">X-API-Key</code> </div> <div style="font-size: 10px; color: #64748b; margin-top: 2px;"> Timing-attack safe HMAC validation </div> </div> """,
            unsafe_allow_html=True,
        )

    # Element 5: Connection Status
    with c_status3:
        if backend_status["online"]:
            status_badge = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.4);">ONLINE (200 OK)</span>'
            status_desc = f"API listening on <strong>{backend_url}</strong> ({backend_status['latency_ms']:.0f}ms latency)"
            border_color = "rgba(16, 185, 129, 0.4)"
        else:
            status_badge = '<span style="background: rgba(244, 63, 94, 0.2); color: #fb7185; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(244, 63, 94, 0.4);">OFFLINE</span>'
            status_desc = f"Cannot reach <code>{backend_url}</code>. Start with <code>python -m src.server</code>"
            border_color = "rgba(244, 63, 94, 0.4)"

        st.markdown(
            f""" <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {border_color}; border-radius: 10px; padding: 14px 16px; min-height: 110px;"> <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"> <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">5. Connection Status</span> {status_badge} </div> <div style="font-size: 13px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;"> Ingestion Pipeline </div> <div style="font-size: 11px; color: #cbd5e1;"> {status_desc} </div> </div> """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Element 3: SDK Installation Command
    # ---------------------------------------------------------------------------
    st.markdown("#### 3. SDK Installation Command")
    st.caption("Install the lightweight, zero-overhead telemetry SDK in your agent project's virtual environment:")
    st.code("pip install agentpulse", language="bash")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Element 4: Integration Example (Exactly matching prompt example)
    # ---------------------------------------------------------------------------
    st.markdown("#### 4. Integration Example Snippet")
    st.caption("Copy this code into your external agent. Telemetry flows automatically through the unified Run/Trace pipeline into this dashboard.")

    example_code_standard = f'''from agentpulse import AgentPulse, observe

# 1. Initialize AgentPulse with your credentials
ap = AgentPulse(
    api_key="{api_key}",
    agent_id="{agent_id}",
    agent_version="{agent_version}",
    backend_url="{backend_url}"
)

# 2. Observe your agent function
@observe(agent_id="{agent_id}", agent_version="{agent_version}")
def my_agent(query: str):
    # Your agent's logic, tool invocations, or LLM calls
    return f"Resolved query: {{query}}"

if __name__ == "__main__":
    result = my_agent("Hello AgentPulse!")
    print(result)
'''

    example_code_tools = f'''import time
from agentpulse import AgentPulse, observe

# Configure SDK
AgentPulse.init(
    backend_url="{backend_url}",
    api_key="{api_key}",
    agent_name="{agent_id}",
    agent_version="{agent_version}"
)

@observe(name="search_kb", span_type="tool")
def search_kb(query: str):
    time.sleep(0.03)
    return {{"status": "found", "info": "Policy details for: " + query}}

@observe(agent_id="{agent_id}", agent_version="{agent_version}", span_type="agent")
def handle_customer_request(user_query: str):
    docs = search_kb(user_query)
    return f"Processed query using doc: {{docs['info']}}"

if __name__ == "__main__":
    ans = handle_customer_request("What is the return window?")
    print("Agent Output:", ans)
    AgentPulse.flush()
'''

    tab_simple, tab_nested = st.tabs(["🚀 Clean Quickstart (Class-based)", "🛠️ With Tool Spans (@observe)"])
    with tab_simple:
        st.code(example_code_standard, language="python")
    with tab_nested:
        st.code(example_code_tools, language="python")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Action: "Test Connection"
    # ---------------------------------------------------------------------------
    st.markdown("#### 🧪 Test Connection")
    st.caption("Verify that your agent credentials and backend URL can successfully ingest telemetry right now.")

    tc_col1, tc_col2 = st.columns([1.5, 3])
    with tc_col1:
        test_clicked = st.button("🧪 Send Test Telemetry Ping", type="primary", width="stretch")

    if test_clicked:
        with st.spinner("Sending synthetic telemetry span to ingestion API..."):
            res = _send_test_span(backend_url, api_key, agent_id, agent_version)
            st.session_state["ca_last_test_result"] = res

    if "ca_last_test_result" in st.session_state:
        res = st.session_state["ca_last_test_result"]
        if res["success"]:
            st.markdown(
                f""" <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 14px 18px; margin-top: 10px;"> <div style="display: flex; align-items: center; justify-content: space-between;"> <div style="display: flex; align-items: center; gap: 8px;"> <span style="font-size: 18px;">✅</span> <div> <div style="font-size: 14px; font-weight: 700; color: #34d399;">Connection Successful! Telemetry Ingested (200 OK)</div> <div style="font-size: 11.5px; color: #cbd5e1;"> Roundtrip latency: <strong>{res['latency_ms']:.1f}ms</strong> · Trace ID: <code>{res['trace_id']}</code> · Run ID: <code>#{res['run_id']}</code> </div> </div> </div> </div> </div> """,
                unsafe_allow_html=True,
            )
            b_col1, b_col2 = st.columns([1.5, 3])
            with b_col1:
                if st.button("🔍 View Ingested Trace in Dashboard ➔", key="btn_ca_view_trace"):
                    navigate_to("🔍 Traces", selected_run_id=res["run_id"])
        else:
            st.markdown(
                f""" <div style="background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 10px; padding: 14px 18px; margin-top: 10px;"> <div style="font-size: 14px; font-weight: 700; color: #fb7185;">Connection Test Failed</div> <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">{res['error']}</div> </div> """,
                unsafe_allow_html=True,
            )
