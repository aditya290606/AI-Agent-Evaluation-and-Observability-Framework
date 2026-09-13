"""
Telemetry data models for AgentPulse SDK.
"""

from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any

from agentpulse.sanitizer import safe_serialize, sanitize_data


@dataclass
class SpanData:
    """Individual span telemetry event captured by @observe."""
    trace_id: str
    span_id: str
    function_name: str
    parent_span_id: Optional[str] = None
    input: Any = ""
    output: Any = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    latency: float = 0.0  # ms
    status: str = "success"  # "success" | "error"
    exception_info: Optional[Dict[str, Any]] = None
    span_type: str = "function"
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    agent_version: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to sanitized dictionary for HTTP transport or local persistence."""
        aid = self.agent_id or self.metadata.get("agent_id") or self.metadata.get("agent_name") or "sdk_agent"
        aver = self.agent_version or self.metadata.get("agent_version") or "1.0.0"
        rid = self.run_id or self.metadata.get("run_id") or self.trace_id
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "run_id": rid,
            "agent_id": aid,
            "agent_version": aver,
            "parent_span_id": self.parent_span_id,
            "function_name": self.function_name,
            "input": safe_serialize(self.input),
            "output": safe_serialize(self.output),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency": round(self.latency, 2),
            "status": self.status,
            "exception_info": sanitize_data(self.exception_info) if self.exception_info else None,
            "span_type": self.span_type,
            "metadata": sanitize_data(self.metadata),
        }

