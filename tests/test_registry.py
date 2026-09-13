"""
Unit tests for the Agent Registry and Agent Adapters.
"""

import os
import sys
import pytest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db
from src.registry.registry import AgentRegistry
from src.registry.adapters import (
    MockAgentAdapter,
    LocalPythonAdapter,
    HttpAgentAdapter,
    AgentExecutionResult,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_registry_seeding_and_list():
    registry = AgentRegistry()
    agents = registry.list_agents()
    assert len(agents) >= 1
    demo = registry.get_agent("demo_react_agent")
    assert demo is not None
    assert demo.name == "Demo ReAct Agent"
    assert demo.integration_type == "mock"


def test_register_new_agent_and_version():
    registry = AgentRegistry()
    unique_name = f"Test Support Bot {uuid.uuid4().hex[:6]}"
    agent = registry.register_agent(
        name=unique_name,
        description="Customer support automated agent",
        framework="crewai",
        provider_model="gpt-4o",
        integration_type="mock",
        initial_version="v1.0",
        config={"inject_bug": False},
    )

    assert agent.name == unique_name
    assert agent.framework == "crewai"
    assert agent.active_version == "v1.0"

    # Add a new version
    v2 = registry.add_version(
        agent_id=agent.agent_id,
        version="v2.0",
        model="gpt-4o-mini",
        config={"inject_bug": True},
        set_active=True,
    )

    assert v2.version == "v2.0"
    updated = registry.get_agent(agent.agent_id)
    assert updated.active_version == "v2.0"


def test_enable_disable_agent():
    registry = AgentRegistry()
    unique_name = f"Toggleable Agent {uuid.uuid4().hex[:6]}"
    agent = registry.register_agent(name=unique_name, integration_type="mock")

    assert agent.status == "active"

    # Disable
    registry.set_agent_status(agent.agent_id, "disabled")
    disabled_agent = registry.get_agent(agent.agent_id)
    assert disabled_agent.status == "disabled"

    # Getting adapter for disabled agent should raise an error
    with pytest.raises(RuntimeError):
        registry.get_adapter(agent.agent_id)

    # Re-enable
    registry.set_agent_status(agent.agent_id, "active")
    re_enabled = registry.get_agent(agent.agent_id)
    assert re_enabled.status == "active"
    adapter = registry.get_adapter(agent.agent_id)
    assert adapter is not None


def test_mock_agent_adapter_execution():
    adapter = MockAgentAdapter(agent_id="mock_1", name="Mock Bot", version="v1.0")
    res = adapter.run("What is 245 multiplied by 18?")

    assert isinstance(res, AgentExecutionResult)
    assert res.success is True
    assert "4410" in res.output
    assert res.trace is not None
    assert len(res.trace.spans) >= 1

    # Test to_base_agent conversion
    base_agent = adapter.to_base_agent()
    from src.core.entities import TestCase
    trace = base_agent.run(TestCase(task_id="T002", query="What is 245 multiplied by 18?"))
    assert "4410" in trace.final_answer


def test_local_python_adapter_execution():
    def dummy_agent_fn(task_id: str, query: str):
        return f"Echo answer for: {query}"

    adapter = LocalPythonAdapter(
        agent_id="py_1",
        name="Python Direct Bot",
        callable_fn=dummy_agent_fn,
    )

    res = adapter.run("How are you?")
    assert res.success is True
    assert "Echo answer for: How are you?" in res.output
    assert len(res.trace.spans) == 1


def test_http_agent_adapter_invalid_url():
    adapter = HttpAgentAdapter(
        agent_id="http_1",
        name="HTTP Bot",
        config={"endpoint_url": ""},
    )

    res = adapter.run("Hello")
    assert res.success is False
    assert "endpoint_url" in res.error
