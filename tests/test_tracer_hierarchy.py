"""
Unit tests for the Hierarchical Observability System and Trace Trees.
"""

import time
import pytest
from src.core.entities import Span
from src.tracing.tracer import RunTrace, build_trace_tree
from src.agent.demo_agent import run_agent_mock


def test_span_types_and_attributes():
    span = Span(
        step_type="retrieval",
        span_id="spn_ret_1",
        parent_span_id="spn_tool_1",
        operation_name="Vector KB Search",
        input_data="query terms",
        output_data="matched document",
        latency_ms=45.2,
        input_tokens=20,
        output_tokens=15,
        model="text-embedding-3-small",
        cost_usd=0.00005,
        attributes={"top_k": 3},
    )

    assert span.span_type == "retrieval"
    assert span.duration_ms == 45.2
    assert span.input == "query terms"
    assert span.output == "matched document"
    assert span.metadata == {"top_k": 3}
    assert span.status == "success"


def test_build_trace_tree_hierarchy():
    root = Span(step_type="agent", span_id="s_root", operation_name="Agent Run")
    llm1 = Span(step_type="llm", span_id="s_llm1", parent_span_id="s_root", operation_name="LLM Decide")
    tool = Span(step_type="tool", span_id="s_tool", parent_span_id="s_root", operation_name="Tool Search")
    ret1 = Span(step_type="retrieval", span_id="s_ret1", parent_span_id="s_tool", operation_name="Doc 1")
    ret2 = Span(step_type="retrieval", span_id="s_ret2", parent_span_id="s_tool", operation_name="Doc 2")
    final = Span(step_type="final_answer", span_id="s_ans", parent_span_id="s_root", operation_name="Answer")

    flat_spans = [root, llm1, tool, ret1, ret2, final]
    tree = build_trace_tree(flat_spans)

    assert len(tree) == 1
    root_node = tree[0]
    assert root_node.span_id == "s_root"
    assert len(root_node.children) == 3  # llm1, tool, final

    tool_node = next(c for c in root_node.children if c.span_id == "s_tool")
    assert len(tool_node.children) == 2  # ret1, ret2


def test_tracer_span_context_manager():
    trace = RunTrace(task_id="T_TEST", query="Test context manager")

    with trace.span("Workflow", span_type="agent") as root_spn:
        root_id = root_spn.span_id
        with trace.span("LLM Call", span_type="llm", parent_span_id=root_id) as child_spn:
            child_spn.output_data = "Ready"

    assert len(trace.spans) == 2
    root = next(s for s in trace.spans if s.step_type == "agent")
    child = next(s for s in trace.spans if s.step_type == "llm")

    assert child.parent_span_id == root.span_id
    assert root.latency_ms >= 0.0
    assert child.status == "success"


def test_demo_agent_mock_hierarchical_spans():
    # Test KB search query which triggers nested retrieval span
    trace = run_agent_mock(task_id="T001", query="How many days does it take to get a refund after a return is received?")

    assert len(trace.spans) >= 4
    span_types = [s.span_type for s in trace.spans]

    assert "agent" in span_types
    assert "llm" in span_types
    assert "tool" in span_types
    assert "retrieval" in span_types
    assert "final_answer" in span_types

    # Test tree reconstruction from the real mock execution
    tree = build_trace_tree(trace.spans)
    assert len(tree) == 1
    root_node = tree[0]
    assert root_node.span_type == "agent"

    # Find the tool span under root
    tool_span = next((s for s in root_node.children if s.span_type == "tool"), None)
    assert tool_span is not None

    # Check nested retrieval under tool span
    retrieval_child = next((s for s in tool_span.children if s.span_type == "retrieval"), None)
    assert retrieval_child is not None
    assert "retrieved" in retrieval_child.output.lower()

    # Check document children under retrieval
    assert len(retrieval_child.children) >= 1
    doc_child = retrieval_child.children[0]
    assert "refund" in doc_child.output.lower() or "days" in doc_child.output.lower()
