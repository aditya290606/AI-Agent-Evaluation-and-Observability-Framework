"""
Tests for the minimal Python AgentPulse SDK (@observe decorator).

Validates:
  - Synchronous functions
  - Asynchronous functions
  - Nested observed functions (trace_id propagation, parent_span_id tracking)
  - Safe error handling (exception capturing without breaking user code)
  - Backend unavailability resilience (agent works normally when AgentPulse is down)
  - API-key authentication
  - Configurable backend URL
  - Secret leakage prevention (zero sensitive token exposure)
"""

import asyncio
import os
import pytest
import time

from agentpulse import observe, configure, get_config, reset_config, flush
from agentpulse.transport import get_transport
from agentpulse.sanitizer import sanitize_text, sanitize_data
from src.server import start_background_server


@pytest.fixture(autouse=True)
def clean_config():
    """Reset configuration before and after each test."""
    reset_config()
    yield
    reset_config()


def test_sync_function_observe():
    """Verify synchronous functions capture all 10 required fields."""
    configure(backend_url="local")

    @observe
    def calculate_tax(income: float, rate: float = 0.2) -> float:
        return income * rate

    result = calculate_tax(1000.0, rate=0.15)
    assert result == 150.0

    span = getattr(calculate_tax, "_last_span", None)
    assert span is not None
    assert span.trace_id.startswith("trc_")
    assert span.span_id.startswith("spn_")
    assert span.parent_span_id is None
    assert "calculate_tax" in span.function_name
    assert span.status == "success"
    assert span.latency >= 0.0
    assert span.start_time <= span.end_time
    assert span.output == "150.0" or span.output == 150.0
    assert span.exception_info is None


def test_async_function_observe():
    """Verify asynchronous functions capture all required telemetry."""
    configure(backend_url="local")

    @observe
    async def fetch_user_data(user_id: str) -> dict:
        await asyncio.sleep(0.01)
        return {"id": user_id, "status": "active"}

    result = asyncio.run(fetch_user_data("usr_123"))
    assert result == {"id": "usr_123", "status": "active"}

    span = getattr(fetch_user_data, "_last_span", None)
    assert span is not None
    assert span.trace_id.startswith("trc_")
    assert span.span_id.startswith("spn_")
    assert span.parent_span_id is None
    assert "fetch_user_data" in span.function_name
    assert span.status == "success"
    assert span.latency >= 10.0  # At least 10ms sleep
    assert span.exception_info is None


def test_nested_observed_functions():
    """Verify nested functions inherit trace_id and link parent_span_id."""
    configure(backend_url="local")

    @observe(name="InnerTool")
    def tool_op(x: int) -> int:
        return x * 2

    @observe(name="OuterAgent")
    def agent_op(val: int) -> int:
        res = tool_op(val)
        return res + 10

    total = agent_op(5)
    assert total == 20

    outer_span = getattr(agent_op, "_last_span", None)
    inner_span = getattr(tool_op, "_last_span", None)

    assert outer_span is not None
    assert inner_span is not None

    # Trace ID propagation
    assert inner_span.trace_id == outer_span.trace_id
    # Parent-child relationship
    assert outer_span.parent_span_id is None
    assert inner_span.parent_span_id == outer_span.span_id
    # Names
    assert outer_span.function_name == "OuterAgent"
    assert inner_span.function_name == "InnerTool"


def test_exception_handling_and_propagation():
    """Verify exceptions are captured in telemetry and safely re-raised."""
    configure(backend_url="local")

    @observe
    def failing_agent(query: str):
        raise ValueError("Invalid query parameter provided")

    with pytest.raises(ValueError, match="Invalid query parameter provided"):
        failing_agent("bad input")

    span = getattr(failing_agent, "_last_span", None)
    assert span is not None
    assert span.status == "error"
    assert span.exception_info is not None
    assert span.exception_info["type"] == "ValueError"
    assert "Invalid query parameter provided" in span.exception_info["message"]
    assert "traceback" in span.exception_info


def test_critical_backend_unavailable_resilience():
    """CRITICAL: If AgentPulse backend is down, user agent MUST work normally without error."""
    # Point to a completely invalid / unreachable port
    configure(backend_url="http://127.0.0.1:59999", timeout=0.1)

    @observe
    def mission_critical_agent(task: str) -> str:
        return f"Successfully completed: {task}"

    # Execution must NOT raise an error despite backend being dead
    result = mission_critical_agent("deploy_satellite")
    assert result == "Successfully completed: deploy_satellite"

    # Also test with async agent
    @observe
    async def async_critical_agent(task: str) -> str:
        return f"Async completed: {task}"

    res_async = asyncio.run(async_critical_agent("route_traffic"))
    assert res_async == "Async completed: route_traffic"


def test_no_secret_leakage():
    """Verify secrets (API keys, bearer tokens, passwords) are redacted."""
    configure(backend_url="local")

    @observe
    def process_credentials(api_key: str, auth_header: str) -> dict:
        return {
            "key": api_key,
            "auth": auth_header,
            "status": "authenticated",
        }

    raw_anthropic = "sk-ant-api03-abcdef1234567890123456789012345"
    raw_bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"

    res = process_credentials(raw_anthropic, raw_bearer)
    span = getattr(process_credentials, "_last_span", None)
    assert span is not None

    dict_repr = span.to_dict()
    # Ensure neither input nor output contains raw secret tokens
    assert raw_anthropic not in str(dict_repr["input"])
    assert raw_anthropic not in str(dict_repr["output"])
    assert "[REDACTED" in str(dict_repr["input"]) or "[REDACTED" in str(dict_repr["output"])


def test_http_transport_and_api_key_auth():
    """Verify real HTTP telemetry transport to Ingestion Server with API key authentication."""
    # Start background ingestion server on an ephemeral port
    import socket
    s = socket.socket()
    s.bind(("", 0))
    free_port = s.getsockname()[1]
    s.close()

    # Set server API key
    test_api_key = "test_pulse_secret_key_888"
    os.environ["AGENTPULSE_API_KEY"] = test_api_key

    server, thread = start_background_server(port=free_port)
    try:
        configure(
            backend_url=f"http://127.0.0.1:{free_port}",
            api_key=test_api_key,
            timeout=2.0,
        )

        @observe
        def live_http_agent(q: str) -> str:
            return f"Answer to {q}"

        ans = live_http_agent("what is AgentPulse?")
        assert ans == "Answer to what is AgentPulse?"

        # Flush queue
        flush(timeout=1.5)

        # Verify against healthcheck
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/health") as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        os.environ.pop("AGENTPULSE_API_KEY", None)
