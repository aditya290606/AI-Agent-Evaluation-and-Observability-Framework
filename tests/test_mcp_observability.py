"""
Comprehensive Test Suite for MCP-Aware Observability & Evaluation.

Tests:
1. Flight recording & Span common interface with MCP attributes
2. Hierarchical Trace Tree (Agent -> MCP Server -> Tool -> Result -> LLM)
3. Evaluator 1: McpToolSelectionEvaluator
4. Evaluator 2: McpArgumentCorrectnessEvaluator
5. Evaluator 3: McpToolSuccessEvaluator
6. Evaluator 4: McpUnnecessaryCallsEvaluator
7. Evaluator 5: McpLatencyEvaluator
8. Evaluator 6: McpFailureAnalysisEvaluator
9. Evaluator 7: McpToolSequenceEvaluator
10. McpEvaluationSuiteEvaluator (Composite Suite)
11. DemoAgent Mock MCP Execution & Trace Logging
"""

import json
import time
import pytest

from src.core.entities import TestCase, Span, Trace
from src.tracing.tracer import RunTrace, build_trace_tree
from src.evaluation.mcp_evaluators import (
    McpToolSelectionEvaluator,
    McpArgumentCorrectnessEvaluator,
    McpToolSuccessEvaluator,
    McpUnnecessaryCallsEvaluator,
    McpLatencyEvaluator,
    McpFailureAnalysisEvaluator,
    McpToolSequenceEvaluator,
    McpEvaluationSuiteEvaluator,
    extract_mcp_telemetry,
    ALL_MCP_EVALUATORS,
)
from src.agent.demo_agent import run_agent_mock


# ===========================================================================
# 1. Span & Flight Recording Tests
# ===========================================================================

def test_mcp_span_properties_and_common_interface():
    """Verify MCP spans preserve the common Span interface while exposing MCP properties."""
    span = Span(
        step_type="tool",
        span_id="spn_1",
        tool_name="search_documents",
        mcp_server="knowledge-base",
        input_data=json.dumps({"query": "refund policy"}),
        output_data=json.dumps({"results": ["30 day refund window"]}),
        latency_ms=120.5,
        status="success",
    )

    assert span.is_mcp is True
    assert span.mcp_server == "knowledge-base"
    assert span.span_type == "mcp_tool"
    assert span.tool_name == "search_documents"
    assert span.duration_ms == 120.5
    assert "refund policy" in span.input
    assert "30 day" in span.output


def test_mcp_flight_recorder_context_manager():
    """Verify RunTrace records MCP spans with context managers and explicit logging."""
    trace = RunTrace(task_id="task_mcp_01", query="Query company knowledge base")

    with trace.span("Agent Planning", span_type="planning") as plan_span:
        plan_span.output_data = "Plan formulated"

    # MCP Server span wrapping an MCP tool call
    with trace.mcp_span(
        tool_name="query_knowledge_base",
        mcp_server="enterprise-kb",
        input_data=json.dumps({"topic": "SOC2 Compliance"}),
    ) as mcp_span:
        mcp_span.output_data = json.dumps({"status": "certified", "year": 2026})

    assert len(trace.spans) == 2
    mcp_recorded = trace.spans[1]
    assert mcp_recorded.is_mcp is True
    assert mcp_recorded.mcp_server == "enterprise-kb"
    assert mcp_recorded.tool_name == "query_knowledge_base"
    assert mcp_recorded.status == "success"
    assert mcp_recorded.duration_ms >= 0.0


# ===========================================================================
# 2. Hierarchical Trace Tree Tests
# ===========================================================================

def test_hierarchical_mcp_trace_tree():
    """Verify hierarchical reconstruction: Agent -> MCP Server -> Tool -> Result -> LLM."""
    spans = [
        Span(
            step_type="agent",
            span_id="spn_root",
            operation_name="Agent Execution",
            parent_span_id=None,
        ),
        Span(
            step_type="mcp_server",
            span_id="spn_srv_1",
            operation_name="MCP Server: Knowledge Base",
            mcp_server="knowledge-base",
            parent_span_id="spn_root",
        ),
        Span(
            step_type="tool",
            span_id="spn_tool_1",
            operation_name="search_documents",
            tool_name="search_documents",
            mcp_server="knowledge-base",
            parent_span_id="spn_srv_1",
            input_data='{"query": "pricing"}',
            output_data='{"price": "$29/mo"}',
        ),
        Span(
            step_type="llm",
            span_id="spn_llm_1",
            operation_name="LLM Synthesis",
            parent_span_id="spn_root",
            output_data="The price is $29/mo.",
        ),
    ]

    roots = build_trace_tree(spans)
    assert len(roots) == 1
    root = roots[0]
    assert root.span_id == "spn_root"
    assert len(root.children) == 2  # MCP Server and LLM

    mcp_server_node = next(c for c in root.children if c.span_id == "spn_srv_1")
    assert len(mcp_server_node.children) == 1
    tool_node = mcp_server_node.children[0]
    assert tool_node.span_id == "spn_tool_1"
    assert tool_node.tool_name == "search_documents"
    assert tool_node.mcp_server == "knowledge-base"


# ===========================================================================
# 3. McpToolSelectionEvaluator Tests
# ===========================================================================

def test_mcp_tool_selection_success():
    """Evaluates correct MCP tool and server selection."""
    tc = TestCase(
        task_id="tc_mcp_sel",
        query="Check SLA",
        expected_mcp_server="sla-service",
        expected_mcp_tools=["get_sla_status"],
    )
    trace = Trace(
        task_id="tc_mcp_sel",
        query="Check SLA",
        spans=[
            Span(
                step_type="tool",
                tool_name="get_sla_status",
                mcp_server="sla-service",
                input_data='{"customer_id": "cust_123"}',
                status="success",
            )
        ],
    )

    evaluator = McpToolSelectionEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace)
    assert res.passed is True
    assert res.score == 1.0
    assert "sla-service" in res.explanation


def test_mcp_tool_selection_mismatch():
    """Evaluates failure when wrong MCP tool or wrong server is called."""
    tc = TestCase(
        task_id="tc_mcp_sel_fail",
        query="Check SLA",
        expected_mcp_server="sla-service",
        expected_mcp_tools=["get_sla_status"],
    )
    trace = Trace(
        task_id="tc_mcp_sel_fail",
        query="Check SLA",
        spans=[
            Span(
                step_type="tool",
                tool_name="get_billing_info",
                mcp_server="wrong-server",
                input_data='{"customer_id": "cust_123"}',
                status="success",
            )
        ],
    )

    evaluator = McpToolSelectionEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace)
    assert res.passed is False
    assert res.score < 1.0
    assert "wrong-server" in res.explanation or "Missing" in res.explanation


# ===========================================================================
# 4. McpArgumentCorrectnessEvaluator Tests
# ===========================================================================

def test_mcp_argument_correctness_pass():
    """Evaluates argument correctness when arguments match expected keys and values."""
    tc = TestCase(
        task_id="tc_mcp_args",
        query="Look up invoice",
        expected_mcp_server="billing-mcp",
        expected_mcp_tools=["get_invoice"],
        expected_arguments={"invoice_id": "INV-2026-001"},
    )
    trace = Trace(
        task_id="tc_mcp_args",
        query="Look up invoice",
        spans=[
            Span(
                step_type="tool",
                tool_name="get_invoice",
                mcp_server="billing-mcp",
                input_data=json.dumps({"invoice_id": "INV-2026-001"}),
                status="success",
            )
        ],
    )

    evaluator = McpArgumentCorrectnessEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace)
    assert res.passed is True
    assert res.score == 1.0


def test_mcp_argument_correctness_fail():
    """Evaluates argument correctness failure on missing required arguments."""
    tc = TestCase(
        task_id="tc_mcp_args_fail",
        query="Look up invoice",
        expected_mcp_server="billing-mcp",
        expected_mcp_tools=["get_invoice"],
        expected_arguments={"invoice_id": "INV-2026-001", "currency": "USD"},
    )
    trace = Trace(
        task_id="tc_mcp_args_fail",
        query="Look up invoice",
        spans=[
            Span(
                step_type="tool",
                tool_name="get_invoice",
                mcp_server="billing-mcp",
                input_data=json.dumps({"invoice_id": "INV-999"}),
                status="success",
            )
        ],
    )

    evaluator = McpArgumentCorrectnessEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace)
    assert res.passed is False
    assert res.score < 1.0


# ===========================================================================
# 5. McpToolSuccessEvaluator Tests
# ===========================================================================

def test_mcp_tool_success_evaluator():
    """Evaluates runtime success across all MCP calls."""
    trace_ok = Trace(
        task_id="tc_ok",
        query="Q",
        spans=[
            Span(
                step_type="tool",
                tool_name="calc",
                mcp_server="math-mcp",
                status="success",
                output_data="42",
            )
        ],
    )
    evaluator = McpToolSuccessEvaluator()
    assert evaluator.evaluate(test_case=TestCase(task_id="tc_ok", query="Q"), trace=trace_ok).passed is True

    trace_err = Trace(
        task_id="tc_err",
        query="Q",
        spans=[
            Span(
                step_type="tool",
                tool_name="calc",
                mcp_server="math-mcp",
                status="error",
                error="DivisionByZeroError: MCP server exception",
            )
        ],
    )
    res_err = evaluator.evaluate(test_case=TestCase(task_id="tc_err", query="Q"), trace=trace_err)
    assert res_err.passed is False
    assert res_err.score == 0.0
    assert "DivisionByZeroError" in res_err.explanation


# ===========================================================================
# 6. McpUnnecessaryCallsEvaluator Tests
# ===========================================================================

def test_mcp_unnecessary_calls_evaluator():
    """Evaluates detection of superfluous or duplicate MCP calls."""
    tc = TestCase(
        task_id="tc_unnec",
        query="Search doc",
        expected_mcp_tools=["search_docs"],
    )
    # 2 identical calls with same args
    trace_dup = Trace(
        task_id="tc_unnec",
        query="Search doc",
        spans=[
            Span(
                step_type="tool",
                tool_name="search_docs",
                mcp_server="docs-mcp",
                input_data='{"q": "auth"}',
                status="success",
            ),
            Span(
                step_type="tool",
                tool_name="search_docs",
                mcp_server="docs-mcp",
                input_data='{"q": "auth"}',
                status="success",
            ),
        ],
    )
    evaluator = McpUnnecessaryCallsEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace_dup)
    assert res.passed is False
    assert res.score < 1.0
    assert "Duplicate" in res.explanation or "unnecessary" in res.explanation


# ===========================================================================
# 7. McpLatencyEvaluator Tests
# ===========================================================================

def test_mcp_latency_evaluator():
    """Evaluates MCP roundtrip latency against latency budgets."""
    tc = TestCase(
        task_id="tc_lat",
        query="Fast fetch",
        mcp_latency_budget=300.0,
    )
    # Fast call within budget
    trace_fast = Trace(
        task_id="tc_lat",
        query="Fast fetch",
        spans=[
            Span(
                step_type="tool",
                tool_name="fetch",
                mcp_server="data-mcp",
                latency_ms=150.0,
                status="success",
            )
        ],
    )
    evaluator = McpLatencyEvaluator()
    res_fast = evaluator.evaluate(test_case=tc, trace=trace_fast)
    assert res_fast.passed is True
    assert res_fast.score == 1.0

    # Slow call exceeding budget
    trace_slow = Trace(
        task_id="tc_lat",
        query="Fast fetch",
        spans=[
            Span(
                step_type="tool",
                tool_name="fetch",
                mcp_server="data-mcp",
                latency_ms=650.0,
                status="success",
            )
        ],
    )
    res_slow = evaluator.evaluate(test_case=tc, trace=trace_slow)
    assert res_slow.passed is False
    assert res_slow.score < 1.0


# ===========================================================================
# 8. McpFailureAnalysisEvaluator Tests
# ===========================================================================

def test_mcp_failure_analysis_evaluator():
    """Evaluates failure categorizations (server disconnect vs tool error vs timeout)."""
    tc = TestCase(task_id="tc_fail_analysis", query="Test resilience")
    trace = Trace(
        task_id="tc_fail_analysis",
        query="Test resilience",
        spans=[
            Span(
                step_type="tool",
                tool_name="api_get",
                mcp_server="remote-mcp",
                status="error",
                error="ConnectionRefusedError: Server connection refused by host",
            ),
            Span(
                step_type="tool",
                tool_name="cache_get",
                mcp_server="local-mcp",
                status="success",
                output_data="cached_val",
            ),
        ],
    )

    evaluator = McpFailureAnalysisEvaluator()
    res = evaluator.evaluate(test_case=tc, trace=trace)
    assert res.score == 0.5
    assert "Connection Error" in str(res.evidence.get("failure_categories")) or "ConnectionRefused" in str(res.evidence)


# ===========================================================================
# 9. McpToolSequenceEvaluator Tests
# ===========================================================================

def test_mcp_tool_sequence_evaluator():
    """Evaluates chronological sequence of MCP tool calls."""
    tc = TestCase(
        task_id="tc_seq",
        query="Authenticate then fetch",
        expected_mcp_tool_sequence=["auth_login", "fetch_profile"],
    )
    trace_correct = Trace(
        task_id="tc_seq",
        query="Authenticate then fetch",
        spans=[
            Span(step_type="tool", tool_name="auth_login", mcp_server="auth-mcp", status="success"),
            Span(step_type="tool", tool_name="fetch_profile", mcp_server="user-mcp", status="success"),
        ],
    )
    evaluator = McpToolSequenceEvaluator()
    res_correct = evaluator.evaluate(test_case=tc, trace=trace_correct)
    assert res_correct.passed is True
    assert res_correct.score == 1.0

    trace_wrong = Trace(
        task_id="tc_seq",
        query="Authenticate then fetch",
        spans=[
            Span(step_type="tool", tool_name="fetch_profile", mcp_server="user-mcp", status="success"),
            Span(step_type="tool", tool_name="auth_login", mcp_server="auth-mcp", status="success"),
        ],
    )
    res_wrong = evaluator.evaluate(test_case=tc, trace=trace_wrong)
    assert res_wrong.passed is False
    assert res_wrong.score < 1.0


# ===========================================================================
# 10. McpEvaluationSuiteEvaluator Tests
# ===========================================================================

def test_mcp_evaluation_suite_composite():
    """Verify composite evaluator runs all 7 sub-evaluators and produces weighted score."""
    tc = TestCase(
        task_id="tc_suite",
        query="Complete MCP workflow",
        expected_mcp_server="kb-server",
        expected_mcp_tools=["search_kb"],
        expected_mcp_tool_sequence=["search_kb"],
        expected_arguments={"query": "agent evaluation"},
        mcp_latency_budget=500.0,
    )
    trace = Trace(
        task_id="tc_suite",
        query="Complete MCP workflow",
        spans=[
            Span(
                step_type="tool",
                tool_name="search_kb",
                mcp_server="kb-server",
                input_data=json.dumps({"query": "agent evaluation"}),
                output_data=json.dumps({"hits": 5}),
                latency_ms=180.0,
                status="success",
            )
        ],
    )

    suite = McpEvaluationSuiteEvaluator()
    res = suite.evaluate(test_case=tc, execution_result="Evaluation completed", trace=trace)
    assert res.metric_name == "mcp_evaluation_suite"
    assert res.passed is True
    assert res.score >= 0.95
    assert "sub_evaluations" in res.evidence
    assert len(res.evidence["sub_evaluations"]) == 7


# ===========================================================================
# 11. DemoAgent Mock MCP Execution Test
# ===========================================================================

def test_demo_agent_mock_mcp_execution():
    """Verify demo_agent runs mock MCP tools and logs structured MCP spans."""
    trace = run_agent_mock("task_mcp_demo", "Search the knowledge base MCP server for security policies", use_mcp=True)
    assert trace is not None
    assert len(trace.spans) >= 3

    mcp_spans = [s for s in trace.spans if s.is_mcp or s.span_type in ("mcp_tool", "mcp_server")]
    assert len(mcp_spans) >= 2

    # Check for presence of MCP server span and MCP tool span
    server_spans = [s for s in trace.spans if s.span_type == "mcp_server"]
    tool_spans = [s for s in trace.spans if s.is_mcp and s.tool_name]
    assert len(server_spans) >= 1
    assert len(tool_spans) >= 1

    tool_span = tool_spans[0]
    assert tool_span.mcp_server is not None
    assert tool_span.input is not None
    assert tool_span.output is not None
    assert tool_span.status == "success"
