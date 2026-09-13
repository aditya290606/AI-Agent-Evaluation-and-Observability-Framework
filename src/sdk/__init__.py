"""
AgentPulse SDK - Developer Interface for Agent Observability & Evaluation.
"""

from src.sdk.client import (
    AgentPulseClient,
    init,
    get_client,
)
from src.sdk.decorators import (
    trace,
    tool,
    mcp_tool,
)
from src.sdk.evaluator import (
    evaluate,
    CallableAgentAdapter,
)

AgentPulse = AgentPulseClient

__all__ = [
    "AgentPulseClient",
    "AgentPulse",
    "init",
    "get_client",
    "trace",
    "tool",
    "mcp_tool",
    "evaluate",
    "CallableAgentAdapter",
]
