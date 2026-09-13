"""Sandboxed Agent Adapter connecting isolated sandbox execution to the Evaluation Framework."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.entities import Span, Trace
from src.registry.adapters import AgentAdapter, AgentExecutionResult
from src.sandbox.manager import SandboxManager
from src.sandbox.models import SandboxConfig, SecurityLevel


class SandboxedAgentAdapter(AgentAdapter):
    """Adapter that executes an uploaded agent project inside a secure sandbox."""

    def __init__(
        self,
        agent_id: str,
        project_dir: Path | str,
        name: str = "Sandboxed Agent",
        version: str = "v1.0",
        entry_point: str = "agent.py",
        entry_symbol: str = "run",
        sandbox_config: Optional[SandboxConfig] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config = dict(config or {})
        merged_config["project_dir"] = str(project_dir)
        merged_config["entry_point"] = entry_point
        merged_config["entry_symbol"] = entry_symbol

        super().__init__(
            agent_id=agent_id,
            name=name,
            version=version,
            config=merged_config,
        )

        self.project_dir = Path(project_dir).resolve()
        self.entry_point = entry_point
        self.entry_symbol = entry_symbol
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.sandbox_manager = sandbox_manager or SandboxManager()

    def run(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        """Execute task inside the isolated sandbox environment and capture full trace."""
        ctx = context or {}
        task_id = ctx.get("task_id", f"sandbox_task_{int(time.time())}")

        start_time = time.time()
        exec_result = self.sandbox_manager.execute_task(
            project_dir=self.project_dir,
            task_id=task_id,
            query=input_text,
            entry_point=self.entry_point,
            entry_symbol=self.entry_symbol,
            config=self.sandbox_config,
        )
        total_latency_ms = (time.time() - start_time) * 1000

        # Construct Trace object
        trace = Trace(
            task_id=task_id,
            query=input_text,
            final_answer=exec_result.response,
            start_time=start_time,
            end_time=start_time + (exec_result.duration_ms / 1000.0),
            metadata={
                "sandbox_security_level": exec_result.security_level.value,
                "sandbox_exit_code": exec_result.exit_code,
                "sandbox_timed_out": exec_result.timed_out,
                "sandbox_status": exec_result.status,
            },
        )

        # Map steps / spans
        if exec_result.trace_steps:
            for idx, raw_step in enumerate(exec_result.trace_steps):
                step_type = raw_step.get("step_type", "agent")
                tool_name = raw_step.get("tool_name")
                span = Span(
                    step_type="tool" if tool_name else (step_type.lower() if step_type else "agent"),
                    tool_name=tool_name,
                    operation_name=f"Sandboxed: {tool_name or step_type}",
                    input_data=raw_step.get("content", input_text),
                    output_data=str(raw_step.get("output", "")),
                    latency_ms=raw_step.get("latency_ms", total_latency_ms / max(1, len(exec_result.trace_steps))),
                    status="error" if raw_step.get("status") in ("FAILED", "ERROR") else "success",
                    step_index=idx + 1,
                    input_tokens=exec_result.token_usage.get("input_tokens", 0) // max(1, len(exec_result.trace_steps)),
                    output_tokens=exec_result.token_usage.get("output_tokens", 0) // max(1, len(exec_result.trace_steps)),
                )
                trace.log_span(span)
        else:
            # Create a single root span for the execution
            span = Span(
                step_type="agent",
                operation_name="Sandboxed Execution",
                input_data=input_text,
                output_data=exec_result.response,
                latency_ms=exec_result.duration_ms,
                status="success" if exec_result.success else "error",
                error=exec_result.error,
                step_index=1,
                input_tokens=exec_result.token_usage.get("input_tokens", 0),
                output_tokens=exec_result.token_usage.get("output_tokens", 0),
            )
            trace.log_span(span)

        return AgentExecutionResult(
            output=exec_result.response,
            trace=trace,
            success=exec_result.success,
            error=exec_result.error,
            latency_ms=exec_result.duration_ms,
            metadata={
                "adapter": "SandboxedAgentAdapter",
                "security_level": exec_result.security_level.value,
                "timed_out": exec_result.timed_out,
                "token_usage": exec_result.token_usage,
            },
        )
