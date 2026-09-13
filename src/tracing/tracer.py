"""
Hierarchical flight recorder and tracer for agent runs.

Supports hierarchical trees of Spans (agent, llm, tool, retrieval, embedding,
memory, planning, final_answer, error), duration timing, token accounting,
cost calculation, and tree reconstruction.
"""

import time
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from src.core.entities import Span, Trace


def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


# Core alias
TracedStep = Span


class RunTrace(Trace):
    """Execution trace managing spans, hierarchical context, and summary stats."""

    def __init__(
        self,
        task_id: str,
        query: str,
        steps: Optional[List[Span]] = None,
        final_answer: str = "",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        is_mock: bool = False,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            task_id=task_id,
            query=query,
            trace_id=trace_id or _gen_id(),
            spans=steps or [],
            final_answer=final_answer,
            start_time=start_time or time.time(),
            end_time=end_time,
            is_mock=is_mock,
            metadata=metadata or {},
        )

    def log_step(self, step: Span):
        from src.security.redactor import SecretRedactor
        from src.security.guardrails import ResourceGuardrails
        SecretRedactor.redact_span(step)
        if step.output_data:
            step.output_data = ResourceGuardrails.truncate_output(step.output_data)
        step.step_index = len(self.spans)
        self.spans.append(step)

    @contextmanager
    def span(
        self,
        operation_name: str,
        span_type: str = "agent",
        parent_span_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        input_data: str = "",
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating, timing, and recording a hierarchical span."""
        from src.security.redactor import SecretRedactor
        from src.security.guardrails import ResourceGuardrails
        s = Span(
            step_type=span_type,
            operation_name=SecretRedactor.redact_text(operation_name),
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            input_data=SecretRedactor.redact_text(input_data),
            model=model,
            attributes=SecretRedactor.redact_data(metadata or {}),
            step_index=len(self.spans),
            start_time=time.time(),
        )
        try:
            yield s
            if not s.status or s.status == "running":
                s.status = "success"
        except Exception as e:
            s.status = "error"
            s.error = SecretRedactor.redact_text(str(e))
            raise
        finally:
            if not s.end_time:
                s.end_time = time.time()
            if not s.latency_ms:
                s.latency_ms = (s.end_time - s.start_time) * 1000
            if s.cost_usd == 0.0:
                from src.cost import global_cost_calculator
                st = (s.step_type or "").lower()
                if "tool" in st or s.tool_name:
                    s.cost_usd = global_cost_calculator.calculate_tool_cost(s.tool_name or s.operation_name)
                elif s.input_tokens > 0 or s.output_tokens > 0:
                    s.cost_usd = global_cost_calculator.calculate_llm_cost(s.model, s.input_tokens, s.output_tokens).total_cost
            SecretRedactor.redact_span(s)
            if s.output_data:
                s.output_data = ResourceGuardrails.truncate_output(s.output_data)
            self.log_span(s)

    @contextmanager
    def mcp_span(
        self,
        server_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        input_data: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        mcp_server: Optional[str] = None,
    ):
        """Context manager for creating, timing, and recording an MCP server or tool execution span."""
        from src.security.redactor import SecretRedactor
        from src.security.guardrails import ResourceGuardrails
        
        effective_server = server_name or mcp_server or "mcp_server"
        span_type = "mcp_tool" if tool_name else "mcp_server"
        op_name = f"MCP Tool: {tool_name} ({effective_server})" if tool_name else f"MCP Server: {effective_server}"
        meta = metadata or {}
        meta["is_mcp"] = True
        meta["mcp_server"] = effective_server
        if tool_name:
            meta["mcp_tool"] = tool_name
        
        s = Span(
            step_type=span_type,
            operation_name=SecretRedactor.redact_text(op_name),
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            mcp_server=effective_server,
            input_data=SecretRedactor.redact_text(str(input_data) if input_data else ""),
            attributes=SecretRedactor.redact_data(meta),
            step_index=len(self.spans),
            start_time=time.time(),
        )
        try:
            yield s
            if not s.status or s.status == "running":
                s.status = "success"
        except Exception as e:
            s.status = "error"
            s.error = SecretRedactor.redact_text(str(e))
            raise
        finally:
            if not s.end_time:
                s.end_time = time.time()
            if not s.latency_ms:
                s.latency_ms = (s.end_time - s.start_time) * 1000
            if s.cost_usd == 0.0:
                from src.cost import global_cost_calculator
                s.cost_usd = global_cost_calculator.calculate_tool_cost(s.tool_name or s.operation_name)
            SecretRedactor.redact_span(s)
            if s.output_data:
                s.output_data = ResourceGuardrails.truncate_output(s.output_data)
            self.log_span(s)

    def log_mcp_call(
        self,
        server_name: Optional[str] = None,
        tool_name: str = "tool",
        input_data: Any = "",
        output_data: Any = "",
        latency_ms: float = 0.0,
        status: str = "success",
        error: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mcp_server: Optional[str] = None,
    ) -> Span:
        """Directly record a completed MCP tool call into the trace."""
        from src.security.redactor import SecretRedactor
        from src.security.guardrails import ResourceGuardrails
        
        effective_server = server_name or mcp_server or "mcp_server"
        meta = metadata or {}
        meta["is_mcp"] = True
        meta["mcp_server"] = effective_server
        meta["mcp_tool"] = tool_name

        s = Span(
            step_type="mcp_tool",
            operation_name=f"MCP Tool: {tool_name} ({server_name})",
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            mcp_server=server_name,
            input_data=SecretRedactor.redact_text(str(input_data) if input_data else ""),
            output_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(output_data) if output_data else "")),
            latency_ms=latency_ms,
            status=status,
            error=SecretRedactor.redact_text(str(error)) if error else None,
            attributes=SecretRedactor.redact_data(meta),
            step_index=len(self.spans),
            start_time=time.time() - (latency_ms / 1000.0) if latency_ms > 0 else time.time(),
            end_time=time.time(),
        )
        self.log_span(s)
        return s


class StepTimer:
    """Context manager for timing a single operation.

    Usage:
        with StepTimer() as t:
            result = do_something()
        trace.log_step(Span(step_type="tool", tool_name="calculator",
                            output_data=result, latency_ms=t.elapsed_ms))
    """

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()

    @property
    def elapsed_ms(self) -> float:
        end = getattr(self, "_end", None) or time.time()
        return (end - self._start) * 1000


def build_trace_tree(spans: List[Span]) -> List[Span]:
    """
    Takes a flat list of Spans and constructs a hierarchical tree using
    span_id and parent_span_id relationships.
    Returns a list of root Spans with their .children populated recursively.
    """
    if not spans:
        return []

    # Map span_id -> Span (resetting children)
    span_map: Dict[str, Span] = {}
    for s in spans:
        s.children = []
        span_map[s.span_id] = s

    roots: List[Span] = []
    for s in spans:
        if s.parent_span_id and s.parent_span_id in span_map and s.parent_span_id != s.span_id:
            span_map[s.parent_span_id].children.append(s)
        else:
            roots.append(s)

    return roots
