"""
The agent being evaluated.

This is intentionally a small ReAct-style agent (LangGraph's prebuilt
create_react_agent) with three tools. It exists to give the evaluation
framework something real to point at. Swap `run_agent_live()` for a call
into YOUR agent (LangGraph, CrewAI, whatever) and everything downstream
(tracing, storage, evaluation, dashboard) keeps working unchanged -- it
only depends on the RunTrace shape, not on how the trace was produced.

Two execution modes:
  - LIVE  : real Claude API calls via langchain-anthropic + LangGraph.
  - MOCK  : deterministic rule-based tool routing, zero API calls/cost.
            Useful to prove the tracing/storage/dashboard plumbing works
            before you spend API budget, and to generate a fast CI check.

`inject_bug=True` deliberately misroutes calculator queries to the
knowledge-base tool in mock mode. It exists purely as a demo device: run
the eval suite once normally (should pass), then again with the bug
injected, and show the framework's regression detection catching it.
"""

import os
import re
import time
from src.agent.tools import ALL_TOOLS, TOOL_MAP, calculator, search_knowledge_base, get_current_date
from src.tracing.tracer import RunTrace, TracedStep, StepTimer, Span


# ---------------------------------------------------------------------------
# MOCK MODE
# ---------------------------------------------------------------------------

def _mock_decide_tool(query: str, inject_bug: bool) -> str:
    q = query.lower()
    if "date" in q or "day of the week" in q:
        return "get_current_date"
    calc_signals = ["multiplied", "divided", "add", "plus", "%", "discount", "full year", "result"]
    if any(sig in q for sig in calc_signals) or re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        return "search_knowledge_base" if inject_bug else "calculator"
    return "search_knowledge_base"


def _mock_extract_calc_expression(query: str) -> str:
    q = query.lower()
    nums = re.findall(r"\d+\.?\d*", query)
    if "multiplied by" in q and len(nums) >= 2:
        return f"{nums[0]} * {nums[1]}"
    if "divided by" in q and len(nums) >= 2:
        return f"{nums[0]} / {nums[1]}"
    if ("add" in q or "plus" in q) and len(nums) >= 2:
        return f"{nums[0]} + {nums[1]}"
    if "%" in query and "discount" in q and len(nums) >= 2:
        return f"{nums[1]} * {float(nums[0]) / 100}"
    if "full year" in q and len(nums) >= 1:
        return f"{nums[0]} * 12"
    if len(nums) >= 2:
        return f"{nums[0]} + {nums[1]}"
    return query


def run_agent_mock(task_id: str, query: str, inject_bug: bool = False, use_mcp: bool = False) -> RunTrace:
    """Deterministic, zero-cost, zero-network execution path with full hierarchical observability and MCP support."""
    trace = RunTrace(task_id=task_id, query=query, is_mock=True)
    root_span_id = f"spn_root_{trace.trace_id[:6]}"

    # Auto-detect MCP from query or flag
    is_mcp_query = use_mcp or ("mcp" in query.lower() or "mcp" in task_id.lower())

    # 1. Root Agent Span
    agent_start = time.time()
    root_span = Span(
        step_type="agent",
        span_id=root_span_id,
        parent_span_id=None,
        operation_name="DemoReActAgent Workflow",
        input_data=query,
        model="mock-deterministic",
        start_time=agent_start,
        status="success",
        attributes={"mode": "mock", "inject_bug": inject_bug, "is_mcp": is_mcp_query},
    )
    trace.log_step(root_span)

    # 2. Planning & Tool Selection (LLM Decision)
    with StepTimer() as t_llm:
        tool_name = _mock_decide_tool(query, inject_bug)
        mcp_tool_name = "search_documents" if tool_name == "search_knowledge_base" else tool_name
        server_name = "Knowledge Base" if tool_name == "search_knowledge_base" else ("Calculator Service" if tool_name == "calculator" else "System Utilities")
    
    llm_decision_span = Span(
        step_type="llm",
        span_id=f"spn_decide_{trace.trace_id[:6]}",
        parent_span_id=root_span_id,
        operation_name="LLM Tool Selection",
        input_data=query,
        output_data=f"Decided to invoke {'MCP tool: ' + mcp_tool_name + ' via server: ' + server_name if is_mcp_query else 'tool: ' + tool_name}",
        latency_ms=t_llm.elapsed_ms,
        start_time=agent_start,
        end_time=agent_start + (t_llm.elapsed_ms / 1000.0),
        input_tokens=len(query.split()),
        output_tokens=5,
        model="mock-haiku",
        status="success",
    )
    trace.log_step(llm_decision_span)

    # 3. Execution Spans (MCP vs Standard)
    if is_mcp_query:
        server_span_id = f"spn_mcp_srv_{trace.trace_id[:6]}"
        server_start = time.time()
        
        # MCP Server Span
        server_span = Span(
            step_type="mcp_server",
            span_id=server_span_id,
            parent_span_id=root_span_id,
            mcp_server=server_name,
            operation_name=f"MCP Server: {server_name}",
            input_data=query,
            status="success",
            start_time=server_start,
            attributes={"protocol": "mcp/1.0", "server_name": server_name, "is_mcp": True},
        )
        trace.log_step(server_span)

        # MCP Tool Execution Span
        tool_span_id = f"spn_mcp_tool_{trace.trace_id[:6]}"
        tool_start = time.time()
        tool_input = query
        with StepTimer() as t_tool:
            from src.agent.tools import _KNOWLEDGE_BASE
            output = search_knowledge_base.invoke(tool_input) if tool_name == "search_knowledge_base" else calculator.invoke(_mock_extract_calc_expression(query))
        
        mcp_tool_span = Span(
            step_type="mcp_tool",
            span_id=tool_span_id,
            parent_span_id=server_span_id,
            tool_name=mcp_tool_name,
            mcp_server=server_name,
            operation_name=f"Tool: {mcp_tool_name}",
            input_data=tool_input,
            output_data=output,
            latency_ms=t_tool.elapsed_ms,
            start_time=tool_start,
            end_time=tool_start + (t_tool.elapsed_ms / 1000.0),
            status="success",
            attributes={"is_mcp": True, "mcp_server": server_name, "mcp_tool": mcp_tool_name},
        )
        trace.log_step(mcp_tool_span)

        server_end = time.time()
        server_span.end_time = server_end
        server_span.latency_ms = (server_end - server_start) * 1000
        server_span.output_data = str(output)

    else:
        # Standard Tool Execution Span
        tool_span_id = f"spn_tool_{trace.trace_id[:6]}"
        tool_start = time.time()

        if tool_name == "calculator":
            tool_input = _mock_extract_calc_expression(query)
            with StepTimer() as t_tool:
                output = calculator.invoke(tool_input)
            tool_span = Span(
                step_type="tool",
                span_id=tool_span_id,
                parent_span_id=root_span_id,
                tool_name=tool_name,
                operation_name=f"Tool: {tool_name}",
                input_data=tool_input,
                output_data=output,
                latency_ms=t_tool.elapsed_ms,
                start_time=tool_start,
                end_time=tool_start + (t_tool.elapsed_ms / 1000.0),
                status="success" if not str(output).startswith("ERROR") else "error",
                error=output if str(output).startswith("ERROR") else None,
                attributes={"expression": tool_input},
            )
            trace.log_step(tool_span)

        elif tool_name == "get_current_date":
            tool_input = ""
            with StepTimer() as t_tool:
                output = get_current_date.invoke(tool_input)
            tool_span = Span(
                step_type="tool",
                span_id=tool_span_id,
                parent_span_id=root_span_id,
                tool_name=tool_name,
                operation_name=f"Tool: {tool_name}",
                input_data=tool_input,
                output_data=output,
                latency_ms=t_tool.elapsed_ms,
                start_time=tool_start,
                end_time=tool_start + (t_tool.elapsed_ms / 1000.0),
                status="success",
            )
            trace.log_step(tool_span)

        else:
            tool_input = query
            with StepTimer() as t_tool:
                from src.agent.tools import _KNOWLEDGE_BASE
                query_terms = set(query.lower().split())
                scored_docs = []
                for doc in _KNOWLEDGE_BASE:
                    score = sum(1 for term in query_terms if term in doc["content"].lower())
                    if score > 0:
                        scored_docs.append((score, doc))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_docs = scored_docs[:2] if scored_docs else ([(1, _KNOWLEDGE_BASE[0])] if _KNOWLEDGE_BASE else [])
                best_doc = top_docs[0][1] if top_docs else None
                output = search_knowledge_base.invoke(tool_input)

                # Normalize relevance scores to [0.0, 1.0] based on max possible
                max_possible_score = len(query_terms) if query_terms else 1
                total_candidates = len(_KNOWLEDGE_BASE)

            tool_span = Span(
                step_type="tool",
                span_id=tool_span_id,
                parent_span_id=root_span_id,
                tool_name=tool_name,
                operation_name=f"Tool: {tool_name}",
                input_data=tool_input,
                output_data=output,
                latency_ms=t_tool.elapsed_ms,
                start_time=tool_start,
                end_time=tool_start + (t_tool.elapsed_ms / 1000.0),
                status="success",
                attributes={
                    "query": tool_input,
                    "retrieval_query": tool_input,
                    "selected_context": output,
                },
            )
            trace.log_step(tool_span)

            if top_docs:
                retrieval_span_id = f"spn_ret_{trace.trace_id[:6]}"
                retrieval_span = Span(
                    step_type="retrieval",
                    span_id=retrieval_span_id,
                    parent_span_id=tool_span_id,
                    operation_name="Retrieval",
                    input_data=tool_input,
                    output_data=f"Retrieved {len(top_docs)} candidate documents",
                    latency_ms=round(t_tool.elapsed_ms * 0.6, 1),
                    start_time=tool_start,
                    end_time=tool_start + (t_tool.elapsed_ms * 0.6 / 1000.0),
                    status="success",
                    attributes={
                        "matched_docs_count": len(top_docs),
                        "retrieval_query": tool_input,
                        "total_candidates": total_candidates,
                        "total_returned": len(top_docs),
                    },
                )
                trace.log_step(retrieval_span)

                for rank, (score, doc) in enumerate(top_docs, 1):
                    normalized_score = round(score / max_possible_score, 4) if max_possible_score > 0 else 0.0
                    doc_span = Span(
                        step_type="retrieval",
                        span_id=f"spn_doc_{trace.trace_id[:6]}_{rank}",
                        parent_span_id=retrieval_span_id,
                        operation_name=f"document {rank}: {doc.get('topic', 'doc')}",
                        input_data=f"query: {tool_input}",
                        output_data=doc["content"],
                        latency_ms=round(t_tool.elapsed_ms * 0.25, 1),
                        start_time=tool_start,
                        end_time=tool_start + (t_tool.elapsed_ms * 0.25 / 1000.0),
                        status="success",
                        attributes={
                            "topic": doc.get("topic"),
                            "doc_id": doc.get("id", f"doc_{rank}"),
                            "relevance_score": normalized_score,
                            "raw_term_overlap": score,
                            "score_provenance": "heuristic",
                        },
                    )
                    trace.log_step(doc_span)

    # 4. Final Answer Formatting & Synthesis
    final_answer = f"Based on {tool_name if not is_mcp_query else mcp_tool_name}: {output}"
    synth_start = time.time()
    synth_span = Span(
        step_type="llm",
        span_id=f"spn_synth_{trace.trace_id[:6]}",
        parent_span_id=root_span_id,
        operation_name="LLM Response Synthesis",
        input_data=f"Tool result: {output}",
        output_data=final_answer,
        latency_ms=1.5,
        start_time=synth_start,
        end_time=synth_start + 0.0015,
        input_tokens=len(output.split()),
        output_tokens=len(final_answer.split()),
        model="mock-haiku",
        status="success",
    )
    trace.log_step(synth_span)

    ans_span = Span(
        step_type="final_answer",
        span_id=f"spn_ans_{trace.trace_id[:6]}",
        parent_span_id=root_span_id,
        operation_name="Final Response Delivery",
        output_data=final_answer,
        status="success",
    )
    trace.log_step(ans_span)

    # Finalize root span
    agent_end = time.time()
    root_span.end_time = agent_end
    root_span.latency_ms = (agent_end - agent_start) * 1000
    root_span.output_data = final_answer

    trace.finish(final_answer)
    return trace


# ---------------------------------------------------------------------------
# LIVE MODE (real Claude API + LangGraph)
# ---------------------------------------------------------------------------

def run_agent_live(task_id: str, query: str) -> RunTrace:
    """Real execution path. Requires ANTHROPIC_API_KEY to be set."""
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model_name = os.getenv("AGENT_MODEL", "claude-3-5-haiku-20241022")
    llm = ChatAnthropic(model=model_name, temperature=0)
    agent = create_react_agent(llm, ALL_TOOLS)

    trace = RunTrace(task_id=task_id, query=query, is_mock=False)

    start = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    messages = result["messages"]

    # Reconstruct step-by-step trace from the message list LangGraph returns.
    step_idx = 0
    pending_tool_args = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = tc.get("id")
                    if call_id:
                        pending_tool_args[call_id] = str(tc.get("args", {}))
                    pending_tool_args[tc["name"]] = str(tc.get("args", {}))
                    trace.log_step(TracedStep(
                        step_type="llm_call",
                        input_data=query if step_idx == 0 else "(reasoning step)",
                        output_data=f"decided to call: {tc['name']} with {tc['args']}",
                        input_tokens=in_tok, output_tokens=out_tok,
                    ))
            elif msg.content:
                trace.log_step(TracedStep(
                    step_type="final_answer",
                    output_data=msg.content if isinstance(msg.content, str) else str(msg.content),
                    input_tokens=in_tok, output_tokens=out_tok,
                ))
        elif isinstance(msg, ToolMessage):
            tool_arg_str = pending_tool_args.get(getattr(msg, "tool_call_id", None)) or pending_tool_args.get(msg.name, "")
            trace.log_step(TracedStep(
                step_type="tool_call",
                tool_name=msg.name,
                input_data=tool_arg_str or "(tool arguments recorded)",
                output_data=str(msg.content),
            ))
        step_idx += 1

    trace.end_time = time.time()
    final_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
    trace.finish(final_msgs[-1].content if final_msgs else "(no final answer produced)")
    return trace


# ---------------------------------------------------------------------------
# Unified entry point & BaseAgent Class Wrapper
# ---------------------------------------------------------------------------

def run_agent(task_id: str, query: str, use_mock: bool, inject_bug: bool = False) -> RunTrace:
    if use_mock:
        return run_agent_mock(task_id, query, inject_bug=inject_bug)
    return run_agent_live(task_id, query)


from src.core.agent_interface import BaseAgent
from src.core.entities import TestCase, Trace


class DemoAgent(BaseAgent):
    """Reference ReAct agent implementing the BaseAgent interface."""

    def __init__(
        self,
        name: str = "DemoReActAgent",
        version: str = "1.0",
        use_mock: bool = True,
        inject_bug: bool = False,
    ):
        super().__init__(
            name=name,
            version=version,
            description="LangGraph ReAct agent with Calculator, Knowledge Base, and Date tools",
        )
        self.use_mock = use_mock
        self.inject_bug = inject_bug
        self.is_mock = use_mock

    def run(self, test_case: TestCase) -> Trace:
        return run_agent(
            task_id=test_case.task_id,
            query=test_case.query,
            use_mock=self.use_mock,
            inject_bug=self.inject_bug,
        )

