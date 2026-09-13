"""
Core client for the AgentPulse SDK.

Provides:
  - AgentPulseClient: Thread-safe, context-aware tracing and telemetry logger.
  - Global client singleton management via init(), get_client().
  - Direct persistence into AgentPulse storage (runs, steps, agents, experiments).
  - Context managers for traces, spans, and MCP calls.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import time
import uuid
from typing import Optional, Dict, Any, List, Union

from src.core.entities import Span, Trace
from src.tracing.tracer import RunTrace
from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, Experiment as ExperimentRow, AgentRecord, AgentVersionRecord
from src.security.redactor import SecretRedactor
from src.security.guardrails import ResourceGuardrails
from src.cost import global_cost_calculator

# Thread/async-safe context tracking
_current_trace_var: ContextVar[Optional[RunTrace]] = ContextVar("_current_trace_var", default=None)
_current_span_var: ContextVar[Optional[Span]] = ContextVar("_current_span_var", default=None)


def _gen_id(prefix: str = "") -> str:
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}_{uid}" if prefix else uid


class AgentPulseClient:
    """AgentPulse SDK Client for external agent observability and tracing."""

    def __init__(
        self,
        agent_name: str = "ExternalAgent",
        agent_version: Optional[str] = None,
        version: Optional[str] = None,
        model: str = "claude-3-5-haiku",
        environment: str = "production",
        persist: bool = True,
        auto_register_agent: bool = True,
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version or version or "v1.0"
        self.model = model
        self.environment = environment
        self.persist = persist
        self.auto_register_agent = auto_register_agent
        init_db()

    @property
    def current_trace(self) -> Optional[RunTrace]:
        """Get the active trace in current execution context."""
        return _current_trace_var.get()

    @property
    def current_span(self) -> Optional[Span]:
        """Get the active parent span in current execution context."""
        return _current_span_var.get()

    @contextmanager
    def trace(
        self,
        name: Optional[str] = None,
        query: str = "",
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_version: Optional[str] = None,
        model: Optional[str] = None,
        experiment_id: Optional[str] = None,
        is_mock: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        persist: Optional[bool] = None,
    ):
        """Context manager to trace an entire agent run.

        Example:
            with client.trace("Customer Support", query="How do I get a refund?") as trace:
                answer = agent.run(query)
                trace.finish(answer)
        """
        tid = task_id or _gen_id("task")
        op_name = name or f"{agent_name or self.agent_name} Run"
        trace_obj = RunTrace(
            task_id=tid,
            query=query,
            is_mock=is_mock,
            metadata=metadata or {},
        )
        trace_obj.metadata["agent_name"] = agent_name or self.agent_name
        trace_obj.metadata["agent_version"] = agent_version or self.agent_version
        trace_obj.metadata["model"] = model or self.model
        trace_obj.metadata["experiment_id"] = experiment_id

        # Root span
        root_span = Span(
            step_type="agent",
            span_id=_gen_id("spn_root"),
            operation_name=op_name,
            input_data=query,
            model=model or self.model,
            start_time=time.time(),
        )
        trace_obj.log_step(root_span)

        token_trace = _current_trace_var.set(trace_obj)
        token_span = _current_span_var.set(root_span)

        should_persist = self.persist if persist is None else persist

        try:
            yield trace_obj
            if not trace_obj.final_answer and root_span.output_data:
                trace_obj.final_answer = root_span.output_data
        except Exception as e:
            root_span.status = "error"
            root_span.error = str(e)
            if not trace_obj.final_answer:
                trace_obj.finish(f"ERROR: {str(e)}")
            raise
        finally:
            if not trace_obj.end_time:
                trace_obj.end_time = time.time()
            if not root_span.end_time:
                root_span.end_time = trace_obj.end_time
                root_span.latency_ms = (root_span.end_time - root_span.start_time) * 1000

            _current_span_var.reset(token_span)
            _current_trace_var.reset(token_trace)

            if should_persist:
                self.persist_trace(trace_obj)

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str = "agent",
        tool_name: Optional[str] = None,
        mcp_server: Optional[str] = None,
        input_data: Any = "",
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager to record a child operation (LLM call, tool, planning, etc.)."""
        current_trace = self.current_trace
        parent_span = self.current_span
        parent_id = parent_span.span_id if parent_span else None

        s = Span(
            step_type=span_type,
            span_id=_gen_id("spn"),
            parent_span_id=parent_id,
            operation_name=SecretRedactor.redact_text(name),
            tool_name=tool_name,
            mcp_server=mcp_server,
            input_data=SecretRedactor.redact_text(str(input_data) if input_data is not None else ""),
            model=model or (parent_span.model if parent_span else None),
            attributes=SecretRedactor.redact_data(metadata or {}),
            start_time=time.time(),
        )

        if current_trace:
            s.step_index = len(current_trace.spans)
            current_trace.log_step(s)

        token_span = _current_span_var.set(s)

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

            # Compute cost if LLM or Tool
            if s.cost_usd == 0.0:
                st = (s.step_type or "").lower()
                if "tool" in st or s.tool_name:
                    s.cost_usd = global_cost_calculator.calculate_tool_cost(s.tool_name or s.operation_name)
                elif s.input_tokens > 0 or s.output_tokens > 0:
                    s.cost_usd = global_cost_calculator.calculate_llm_cost(s.model, s.input_tokens, s.output_tokens).total_cost

            if s.output_data:
                s.output_data = ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(s.output_data)))

            _current_span_var.reset(token_span)

    @contextmanager
    def mcp_span(
        self,
        server_name: str,
        tool_name: Optional[str] = None,
        input_data: Any = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager specifically for Model Context Protocol (MCP) server or tool execution."""
        op_name = f"MCP Tool: {tool_name} ({server_name})" if tool_name else f"MCP Server: {server_name}"
        meta = metadata or {}
        meta["is_mcp"] = True
        meta["mcp_server"] = server_name
        if tool_name:
            meta["mcp_tool"] = tool_name

        with self.span(
            name=op_name,
            span_type="mcp_tool" if tool_name else "mcp_server",
            tool_name=tool_name,
            mcp_server=server_name,
            input_data=input_data,
            metadata=meta,
        ) as s:
            yield s

    def log_tool_call(
        self,
        tool_name: str,
        input_data: Any = "",
        output_data: Any = "",
        latency_ms: float = 0.0,
        status: str = "success",
        error: Optional[str] = None,
        is_mcp: bool = False,
        mcp_server: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Record a completed tool call into the currently active trace."""
        trace_obj = self.current_trace
        if not trace_obj:
            return None

        parent_span = self.current_span
        parent_id = parent_span.span_id if parent_span else None

        meta = metadata or {}
        if is_mcp or mcp_server:
            meta["is_mcp"] = True
            if mcp_server:
                meta["mcp_server"] = mcp_server
            meta["mcp_tool"] = tool_name

        step_type = "mcp_tool" if (is_mcp or mcp_server) else "tool_call"
        op_name = f"MCP Tool: {tool_name} ({mcp_server})" if mcp_server else f"Tool: {tool_name}"

        span = Span(
            step_type=step_type,
            span_id=_gen_id("spn_tool"),
            parent_span_id=parent_id,
            operation_name=op_name,
            tool_name=tool_name,
            mcp_server=mcp_server,
            input_data=SecretRedactor.redact_text(str(input_data) if input_data is not None else ""),
            output_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(output_data) if output_data is not None else "")),
            latency_ms=latency_ms,
            status=status,
            error=SecretRedactor.redact_text(str(error)) if error else None,
            attributes=SecretRedactor.redact_data(meta),
            cost_usd=global_cost_calculator.calculate_tool_cost(tool_name),
            step_index=len(trace_obj.spans),
            start_time=time.time() - (latency_ms / 1000.0) if latency_ms > 0 else time.time(),
            end_time=time.time(),
        )
        trace_obj.log_step(span)
        return span

    def log_mcp_call(
        self,
        server_name: str,
        tool_name: str,
        input_data: Any = "",
        output_data: Any = "",
        latency_ms: float = 0.0,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Convenience method to record an MCP call into the active trace."""
        return self.log_tool_call(
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            status=status,
            error=error,
            is_mcp=True,
            mcp_server=server_name,
            metadata=metadata,
        )

    def persist_trace(
        self,
        trace: Union[RunTrace, Trace],
        experiment_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_version: Optional[str] = None,
        model: Optional[str] = None,
        environment: Optional[str] = None,
        expected_tool: Optional[str] = None,
    ) -> Run:
        """Persist an executed trace and its spans into SQLite/PostgreSQL storage."""
        session = get_session()
        try:
            eff_agent = agent_name or trace.metadata.get("agent_name") or self.agent_name
            eff_version = agent_version or trace.metadata.get("agent_version") or self.agent_version
            eff_model = model or trace.metadata.get("model") or self.model
            eff_env = environment or trace.metadata.get("environment") or self.environment
            eff_exp = experiment_id or trace.metadata.get("experiment_id") or "sdk_runs"

            # Auto-register agent in registry if configured
            if self.auto_register_agent:
                self._ensure_agent_registered(session, eff_agent, eff_version, eff_model)

            # Ensure default experiment exists if needed
            self._ensure_experiment_exists(session, eff_exp, eff_agent, eff_version, eff_model, eff_env)

            # Calculate token and cost estimates
            total_cost = sum(s.cost_usd for s in trace.spans)
            actual_cost = trace.metadata.get("actual_cost")

            tools_called_list = trace.tools_called
            if not tools_called_list:
                tools_called_list = [s.tool_name for s in trace.spans if s.tool_name]

            run_row = Run(
                experiment_id=eff_exp,
                agent_name=eff_agent,
                agent_id=eff_agent.lower().replace(" ", "_"),
                agent_version=eff_version,
                model=eff_model,
                prompt_version=trace.metadata.get("prompt_version", "v1.0"),
                dataset_version=trace.metadata.get("dataset_version", "v1.0"),
                environment=eff_env,
                task_id=trace.task_id or _gen_id("task"),
                query=SecretRedactor.redact_text(trace.query or ""),
                final_answer=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(trace.final_answer or "")),
                expected_tool=expected_tool or trace.metadata.get("expected_tool", ""),
                tools_called=",".join(tools_called_list),
                total_input_tokens=trace.total_input_tokens,
                total_output_tokens=trace.total_output_tokens,
                latency_ms=trace.latency_ms,
                est_cost_usd=total_cost,
                actual_cost_usd=actual_cost,
                is_mock=trace.is_mock,
            )
            session.add(run_row)
            session.flush()

            # Record steps/spans
            for idx, s in enumerate(trace.spans):
                session.add(Step(
                    run_id=run_row.id,
                    step_index=s.step_index or idx,
                    step_type=s.step_type or "step",
                    span_id=s.span_id or _gen_id("spn"),
                    parent_span_id=s.parent_span_id,
                    operation_name=s.operation_name or s.name,
                    tool_name=s.tool_name,
                    input_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(s.input_data or "")),
                    output_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(s.output_data or "")),
                    latency_ms=s.latency_ms,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    status=s.status,
                    error=SecretRedactor.redact_text(s.error) if s.error else None,
                    model=s.model or eff_model,
                    cost_usd=s.cost_usd,
                    metadata_json=json.dumps(SecretRedactor.redact_data(s.attributes or {})),
                    input_tokens=s.input_tokens,
                    output_tokens=s.output_tokens,
                ))

            session.commit()
            return run_row
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_run(
        self,
        query: str,
        final_answer: str,
        tools_called: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_version: Optional[str] = None,
        model: Optional[str] = None,
        spans: Optional[List[Span]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """One-line helper to directly log an external agent run into AgentPulse."""
        trace = RunTrace(
            task_id=task_id or _gen_id("task"),
            query=query,
            final_answer=final_answer,
            steps=spans or [],
            metadata=metadata or {},
        )
        trace.finish(final_answer)
        if latency_ms > 0:
            trace.latency_ms = latency_ms

        if tools_called and not spans:
            for t in tools_called:
                trace.log_step(Span(step_type="tool_call", tool_name=t, output_data="Executed"))

        return self.persist_trace(
            trace=trace,
            agent_name=agent_name,
            agent_version=agent_version,
            model=model,
        )

    def _ensure_agent_registered(self, session, agent_name: str, agent_version: str, model: str):
        agent_id = agent_name.lower().replace(" ", "_")
        existing_agent = session.query(AgentRecord).filter(AgentRecord.agent_id == agent_id).first()
        if not existing_agent:
            agent_rec = AgentRecord(
                agent_id=agent_id,
                name=agent_name,
                description=f"External agent registered via AgentPulse SDK",
                framework="sdk",
                provider_model=model,
                integration_type="sdk",
                status="active",
                active_version=agent_version,
            )
            session.add(agent_rec)
            session.flush()

        # Check version
        ver_id = f"ver_{agent_id}_{agent_version.replace('.', '_')}"
        existing_ver = session.query(AgentVersionRecord).filter(AgentVersionRecord.version_id == ver_id).first()
        if not existing_ver:
            ver_rec = AgentVersionRecord(
                version_id=ver_id,
                agent_id=agent_id,
                version=agent_version,
                model=model,
                configuration_metadata=json.dumps({"source": "agentpulse_sdk"}),
                status="active",
            )
            session.add(ver_rec)
            session.flush()

    def _ensure_experiment_exists(self, session, experiment_id: str, agent_name: str, agent_version: str, model: str, environment: str):
        existing_exp = session.query(ExperimentRow).filter(ExperimentRow.id == experiment_id).first()
        if not existing_exp:
            exp = ExperimentRow(
                id=experiment_id,
                name=f"SDK Runs ({agent_name})",
                description="Live and external evaluation traces logged via AgentPulse SDK",
                agent_id=agent_name.lower().replace(" ", "_"),
                agent_name=agent_name,
                agent_version=agent_version,
                model=model,
                prompt_version="v1.0",
                dataset_name="external_live",
                dataset_version="v1.0",
                environment=environment,
                status="completed",
            )
            session.add(exp)
            session.flush()


# Global default client
_global_client: Optional[AgentPulseClient] = None


def init(
    agent_name: str = "ExternalAgent",
    agent_version: Optional[str] = None,
    version: Optional[str] = None,
    model: str = "claude-3-5-haiku",
    environment: str = "production",
    persist: bool = True,
) -> AgentPulseClient:
    """Initialize the global AgentPulse SDK client."""
    global _global_client
    _global_client = AgentPulseClient(
        agent_name=agent_name,
        agent_version=agent_version or version or "v1.0",
        model=model,
        environment=environment,
        persist=persist,
    )
    return _global_client


def get_client() -> AgentPulseClient:
    """Retrieve or automatically create the global AgentPulse client."""
    global _global_client
    if _global_client is None:
        _global_client = AgentPulseClient()
    return _global_client
