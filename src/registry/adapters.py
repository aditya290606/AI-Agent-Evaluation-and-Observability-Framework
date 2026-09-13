"""
Agent Adapter interfaces and concrete implementations for the Agent Registry.

Supported integration types:
  1. MockAgentAdapter (internal deterministic / demo agent)
  2. LocalPythonAdapter (callable or module path)
  3. HttpAgentAdapter (REST / HTTP endpoint)

All adapters implement a unified contract:
  run(input_text, context) -> AgentExecutionResult
and can be converted to BaseAgent via to_base_agent() for the EvaluationEngine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
import json
import time
from typing import Optional, Dict, Any, Callable
import requests

from src.core.entities import Trace, Span, TestCase
from src.core.agent_interface import BaseAgent
from src.tracing.tracer import RunTrace, TracedStep, StepTimer


@dataclass
class AgentExecutionResult:
    """Unified execution result returned by all AgentAdapters."""
    output: str
    trace: Trace
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """Abstract base class for all agent adapters."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        version: str = "v1.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.version = version
        self.config = config or {}

    @abstractmethod
    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        """Execute the agent on the given input with optional contextual metadata."""
        pass

    def to_base_agent(self) -> BaseAgent:
        """Wrap this adapter into a BaseAgent for direct use in the EvaluationEngine."""
        adapter = self

        class _AdapterBaseAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    name=adapter.name,
                    version=adapter.version,
                    description=f"Adapter for {adapter.agent_id} ({adapter.__class__.__name__})",
                    config=adapter.config,
                )
                self.agent_id = adapter.agent_id

            def run(self, test_case: TestCase) -> Trace:
                res = adapter.run(
                    test_case.query,
                    context={"task_id": test_case.task_id, "metadata": test_case.metadata},
                )
                return res.trace

        return _AdapterBaseAgent()


# ---------------------------------------------------------------------------
# 1. Mock Agent Adapter (Internal baseline demo agent)
# ---------------------------------------------------------------------------

class MockAgentAdapter(AgentAdapter):
    """Adapter for internal mock/rule-based agents."""

    def __init__(
        self,
        agent_id: str = "demo_react_agent",
        name: str = "Demo ReAct Agent",
        version: str = "v1.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id=agent_id, name=name, version=version, config=config)
        self.inject_bug = self.config.get("inject_bug", False)

    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        from src.agent.demo_agent import run_agent_mock
        ctx = context or {}
        task_id = ctx.get("task_id", "mock_task")

        start = time.time()
        trace = run_agent_mock(task_id=task_id, query=input_text, inject_bug=self.inject_bug)
        elapsed_ms = (time.time() - start) * 1000

        return AgentExecutionResult(
            output=trace.final_answer,
            trace=trace,
            success=True,
            latency_ms=elapsed_ms,
            metadata={"adapter": "MockAgentAdapter", "inject_bug": self.inject_bug},
        )


# ---------------------------------------------------------------------------
# 2. Local Python Adapter
# ---------------------------------------------------------------------------

class LocalPythonAdapter(AgentAdapter):
    """Adapter for executing a local Python callable or module path.

    Config options:
      - module_path: e.g. "src.agent.custom_agent_template"
      - callable_name: e.g. "run_custom_agent"
      - callable: direct Python function reference (optional)
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        version: str = "v1.0",
        config: Optional[Dict[str, Any]] = None,
        callable_fn: Optional[Callable] = None,
    ):
        super().__init__(agent_id=agent_id, name=name, version=version, config=config)
        self._callable_fn = callable_fn

    def _resolve_callable(self) -> Callable:
        if self._callable_fn:
            return self._callable_fn

        module_path = self.config.get("module_path")
        callable_name = self.config.get("callable_name", "run_agent")

        if not module_path:
            raise ValueError(f"LocalPythonAdapter requires 'module_path' in config, got {self.config}")

        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, callable_name)
            return fn
        except Exception as e:
            raise ImportError(f"Failed to import {callable_name} from {module_path}: {e}")

    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        ctx = context or {}
        task_id = ctx.get("task_id", "adhoc_task")
        trace = Trace(task_id=task_id, query=input_text)
        start = time.time()

        try:
            target_fn = self._resolve_callable()
            with StepTimer() as timer:
                # Target function may accept (task_id, query) or just (query)
                import inspect
                sig = inspect.signature(target_fn)
                if len(sig.parameters) >= 2:
                    raw_result = target_fn(task_id, input_text)
                else:
                    raw_result = target_fn(input_text)

            # If the function returned a Trace/RunTrace object directly, use it
            if isinstance(raw_result, (Trace, RunTrace)):
                trace = raw_result
                output = trace.final_answer
            elif getattr(target_fn, "_last_trace", None) is not None:
                trace = target_fn._last_trace
                output = trace.final_answer or str(raw_result)
            else:
                output = str(raw_result)
                trace.log_span(Span(
                    step_type="llm_call",
                    name="python_callable",
                    input_data=input_text,
                    output_data=output,
                    latency_ms=timer.elapsed_ms,
                ))
                trace.finish(output)

            elapsed_ms = (time.time() - start) * 1000
            return AgentExecutionResult(
                output=output,
                trace=trace,
                success=True,
                latency_ms=elapsed_ms,
                metadata={"adapter": "LocalPythonAdapter"},
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            err_msg = f"LocalPythonAdapter execution error: {str(e)}"
            trace.log_span(Span(
                step_type="error",
                name="python_exception",
                output_data=err_msg,
                latency_ms=elapsed_ms,
            ))
            trace.finish(f"ERROR: {err_msg}")
            return AgentExecutionResult(
                output=err_msg,
                trace=trace,
                success=False,
                error=err_msg,
                latency_ms=elapsed_ms,
            )


# ---------------------------------------------------------------------------
# 3. HTTP / API Agent Adapter
# ---------------------------------------------------------------------------

class HttpAgentAdapter(AgentAdapter):
    """Adapter for querying an external HTTP REST API agent.

    Config options:
      - endpoint_url: "https://api.example.com/v1/agent" (required)
      - method: "POST" (default)
      - headers: {"Authorization": "Bearer ...", ...}
      - timeout: 30.0 (seconds)
      - input_key: JSON field name for the query, default "query" or "input"
      - output_key: JSON field name to extract the final answer from, default "output" or "answer"
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        version: str = "v1.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id=agent_id, name=name, version=version, config=config)
        self.endpoint_url = self.config.get("endpoint_url", "")
        self.timeout = float(self.config.get("timeout", 30.0))
        self.headers = self.config.get("headers", {})
        self.input_key = self.config.get("input_key", "query")
        self.output_key = self.config.get("output_key", "answer")

    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        ctx = context or {}
        task_id = ctx.get("task_id", "http_task")
        trace = Trace(task_id=task_id, query=input_text)
        start = time.time()

        if not self.endpoint_url:
            err_msg = "HttpAgentAdapter error: 'endpoint_url' is not configured."
            trace.log_span(Span(step_type="error", output_data=err_msg))
            trace.finish(err_msg)
            return AgentExecutionResult(
                output=err_msg,
                trace=trace,
                success=False,
                error=err_msg,
                latency_ms=0.0,
            )

        payload = {
            self.input_key: input_text,
            "task_id": task_id,
            "context": ctx,
        }

        try:
            from src.security.redactor import SecretRedactor
            from src.security.guardrails import ResourceGuardrails

            req_headers = {"Content-Type": "application/json"}
            req_headers.update(self.headers)

            resp = requests.post(
                self.endpoint_url,
                json=payload,
                headers=req_headers,
                timeout=self.timeout,
            )
            elapsed_ms = (time.time() - start) * 1000

            if resp.status_code >= 400:
                err_msg = f"HTTP {resp.status_code} from {self.endpoint_url}: {SecretRedactor.redact_text(resp.text[:200])}"
                trace.log_span(Span(
                    step_type="http_call",
                    tool_name="http_post",
                    input_data=SecretRedactor.redact_text(json.dumps(payload)),
                    output_data=err_msg,
                    latency_ms=elapsed_ms,
                ))
                trace.finish(err_msg)
                trace.sanitize()
                return AgentExecutionResult(
                    output=err_msg,
                    trace=trace,
                    success=False,
                    error=err_msg,
                    latency_ms=elapsed_ms,
                )

            # Try parsing JSON response
            try:
                data = resp.json()
                if isinstance(data, dict):
                    output = (
                        data.get(self.output_key)
                        or data.get("output")
                        or data.get("response")
                        or data.get("answer")
                        or str(data)
                    )
                    # If response includes tool calls or trace info, extract spans
                    tools_used = data.get("tools_called", [])
                    if isinstance(tools_used, list):
                        for tool in tools_used:
                            trace.log_span(Span(
                                step_type="tool_call",
                                tool_name=str(tool),
                                output_data="Executed via remote HTTP agent",
                            ))
                    # Check for actual cost in payload or headers
                    raw_cost = (
                        data.get("actual_cost")
                        or data.get("actual_cost_usd")
                        or data.get("cost_usd")
                        or (data.get("usage", {}).get("total_cost") if isinstance(data.get("usage"), dict) else None)
                        or resp.headers.get("x-actual-cost")
                    )
                    if raw_cost is not None:
                        try:
                            trace.metadata["actual_cost"] = float(raw_cost)
                        except (ValueError, TypeError):
                            pass
                else:
                    output = str(data)
            except Exception:
                output = resp.text

            sanitized_output = ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(output)))

            trace.log_span(Span(
                step_type="http_call",
                name="http_agent_endpoint",
                input_data=SecretRedactor.redact_text(json.dumps(payload)),
                output_data=sanitized_output,
                latency_ms=elapsed_ms,
            ))
            trace.finish(sanitized_output)
            trace.sanitize()

            return AgentExecutionResult(
                output=sanitized_output,
                trace=trace,
                success=True,
                latency_ms=elapsed_ms,
                metadata={"status_code": resp.status_code, "adapter": "HttpAgentAdapter", "actual_cost": trace.metadata.get("actual_cost")},
            )

        except Exception as e:
            from src.security.redactor import SecretRedactor
            elapsed_ms = (time.time() - start) * 1000
            err_msg = SecretRedactor.redact_text(f"HttpAgentAdapter request failed: {str(e)}")
            trace.log_span(Span(
                step_type="error",
                name="http_connection_error",
                output_data=err_msg,
                latency_ms=elapsed_ms,
            ))
            trace.finish(err_msg)
            trace.sanitize()
            return AgentExecutionResult(
                output=err_msg,
                trace=trace,
                success=False,
                error=err_msg,
                latency_ms=elapsed_ms,
            )



# ---------------------------------------------------------------------------
# 4. Callable Agent Adapter (Direct Python Function / SDK Bridge)
# ---------------------------------------------------------------------------

class CallableAgentAdapter(AgentAdapter):
    """Adapter bridging arbitrary Python callables into the Evaluation framework."""

    def __init__(
        self,
        fn: Callable,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        version: str = "v1.0",
        model: str = "claude-3-5-haiku",
    ):
        resolved_name = name or getattr(fn, "__name__", "CustomCallableAgent")
        resolved_id = agent_id or resolved_name.lower().replace(" ", "_")
        super().__init__(agent_id=resolved_id, name=resolved_name, version=version)
        self.fn = fn
        self.model = model

        # Ensure registered in AgentRecord as SDK integration
        try:
            from src.storage.db import get_session
            from src.storage.models import AgentRecord
            session = get_session()
            try:
                rec = session.query(AgentRecord).filter_by(agent_id=resolved_id).first()
                if not rec:
                    rec = AgentRecord(
                        agent_id=resolved_id,
                        name=resolved_name,
                        description=f"External callable agent registered via SDK",
                        framework="sdk",
                        provider_model=self.model,
                        integration_type="sdk",
                        status="active",
                        active_version=version,
                    )
                    session.add(rec)
                    session.commit()
                elif rec.integration_type != "sdk":
                    rec.integration_type = "sdk"
                    session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()
        except Exception:
            pass

    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        ctx = context or {}
        task_id = ctx.get("task_id", "adhoc_task")
        from src.sdk.client import get_client
        client = get_client()

        with client.trace(
            name=f"{self.name} Eval Run",
            query=input_text,
            task_id=task_id,
            agent_name=self.name,
            agent_version=self.version,
            model=self.model,
            persist=False,
        ) as t:
            try:
                import inspect
                sig = inspect.signature(self.fn)
                if len(sig.parameters) >= 2:
                    raw_res = self.fn(task_id, input_text)
                else:
                    raw_res = self.fn(input_text)

                if isinstance(raw_res, (RunTrace, Trace)):
                    t.final_answer = raw_res.final_answer
                    if raw_res.spans:
                        existing_ids = {s.span_id for s in t.spans if getattr(s, "span_id", None)}
                        for s in raw_res.spans:
                            if s.span_id not in existing_ids:
                                t.log_step(s)
                    output = raw_res.final_answer
                elif getattr(self.fn, "_last_trace", None) is not None:
                    last_t = getattr(self.fn, "_last_trace")
                    t.final_answer = last_t.final_answer
                    if last_t.spans:
                        existing_ids = {s.span_id for s in t.spans if getattr(s, "span_id", None)}
                        for s in last_t.spans:
                            if s.span_id not in existing_ids:
                                t.log_step(s)
                    output = t.final_answer or str(raw_res)
                else:
                    output = str(raw_res) if raw_res is not None else ""
                    t.finish(output)

                return AgentExecutionResult(
                    output=output,
                    trace=t,
                    success=True,
                    latency_ms=t.latency_ms,
                    metadata={"adapter": "CallableAgentAdapter"},
                )
            except Exception as e:
                err_msg = f"Error in external agent function: {str(e)}"
                t.finish(f"ERROR: {err_msg}")
                return AgentExecutionResult(
                    output=err_msg,
                    trace=t,
                    success=False,
                    error=err_msg,
                    latency_ms=t.latency_ms,
                )
