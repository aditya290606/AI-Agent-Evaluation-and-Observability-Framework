"""
Unit tests for the AgentPulse SDK.

Tests:
  - Client initialization and context tracking
  - Trace, span, and mcp_span context managers
  - @trace, @tool, and @mcp_tool decorators
  - Trace persistence into storage (runs, steps, agents)
  - CallableAgentAdapter bridging
  - Programmatic agentpulse.evaluate() workflow
"""

import os
import pytest
import time
from typing import Dict, Any

import agentpulse
from agentpulse import AgentPulseClient, trace, tool, mcp_tool, evaluate
from src.core.entities import TestCase, EvaluationDataset
from src.storage.db import get_session
from src.storage.models import Run, Step, AgentRecord


def test_sdk_init_and_client_singleton():
    client = agentpulse.init(
        agent_name="TestBot",
        agent_version="v0.9",
        model="mock-haiku",
        persist=False,
    )
    assert client is not None
    assert client.agent_name == "TestBot"
    assert client.agent_version == "v0.9"
    assert agentpulse.get_client() is client


def test_context_manager_tracing():
    client = AgentPulseClient(agent_name="CtxBot", persist=False)

    with client.trace("Test Context Workflow", query="what is 2+2?", task_id="test_ctx_01") as t:
        assert client.current_trace is t
        assert len(t.spans) == 1  # Root span created

        with client.span("Sub Operation", span_type="planning") as s_plan:
            assert client.current_span is s_plan
            s_plan.output_data = "Plan: compute 2+2"

        with client.mcp_span("CalculatorServer", tool_name="add") as s_mcp:
            s_mcp.output_data = "4"
            assert s_mcp.is_mcp is True
            assert s_mcp.mcp_server == "CalculatorServer"

        t.finish("4")

    assert t.final_answer == "4"
    assert len(t.spans) == 3
    assert client.current_trace is None


def test_decorators_end_to_end():
    client = agentpulse.init(agent_name="DecoratorBot", version="v1.0", persist=False)

    @tool(name="unit_calc")
    def do_calc(expr: str) -> str:
        return str(eval(expr))

    @mcp_tool(server="DocServer", name="lookup")
    def do_lookup(term: str) -> str:
        return f"Documentation for {term}"

    @trace(agent_name="DecoratorBot", version="v1.0", persist=False)
    def my_agent(query: str) -> str:
        if "calc" in query:
            return do_calc("5 * 5")
        return do_lookup("refund")

    ans = my_agent("calc: compute 5*5")
    assert ans == "25"
    last_trace = getattr(my_agent, "_last_trace", None)
    assert last_trace is not None
    assert last_trace.final_answer == "25"

    tool_spans = [s for s in last_trace.spans if s.tool_name == "unit_calc"]
    assert len(tool_spans) == 1
    assert tool_spans[0].output_data == "25"


def test_sdk_persistence_to_db():
    client = AgentPulseClient(agent_name="PersistedBot", agent_version="v3.1", persist=True)

    with client.trace(
        name="DB Persistence Test",
        query="Persist me into SQLite",
        task_id="task_persist_99",
        metadata={"test": True},
    ) as t:
        client.log_tool_call("test_tool", input_data="ping", output_data="pong", latency_ms=15.0)
        client.log_mcp_call("TestMCP", "mcp_tool", input_data="foo", output_data="bar", latency_ms=25.0)
        t.finish("Done persisting")

    # Verify run and steps in database
    session = get_session()
    try:
        run_row = session.query(Run).filter(Run.task_id == "task_persist_99").first()
        assert run_row is not None
        assert run_row.agent_name == "PersistedBot"
        assert run_row.agent_version == "v3.1"
        assert run_row.final_answer == "Done persisting"

        steps = session.query(Step).filter(Step.run_id == run_row.id).all()
        assert len(steps) >= 3  # Root agent + test_tool + mcp_tool

        agent_rec = session.query(AgentRecord).filter(AgentRecord.agent_id == "persistedbot").first()
        assert agent_rec is not None
    finally:
        session.close()


def test_sdk_evaluate_bridge():
    @tool(name="calculator")
    def calc_fn(expr: str) -> str:
        return "18"

    @trace(agent_name="EvalCandidateAgent", version="v1.0", persist=False)
    def simple_agent(query: str) -> str:
        calc_fn("120 * 0.15")
        return "The calculated 15% tip is 18 dollars."

    custom_test = TestCase(
        test_id="SDK_T01",
        user_input="Calculate 15% tip on 120",
        expected_tools=["calculator"],
        expected_keywords=["18"],
        latency_budget=3000.0,
    )
    dataset = EvaluationDataset(name="sdk_eval_test", test_cases=[custom_test])

    report = evaluate(
        agent=simple_agent,
        dataset=dataset,
        experiment_name="sdk_unit_eval",
        persist=False,
        print_summary=False,
    )

    assert report is not None
    assert report.total_test_cases == 1
    assert report.passed_checks > 0
    assert report.overall_pass_rate > 50.0
