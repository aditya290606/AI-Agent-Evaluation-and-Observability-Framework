"""
AgentPulse - Full-Stack AI Agent Observability & Evaluation Framework SDK.

Developer Experience:
    pip install agentpulse

Example:
    from agentpulse import observe

    @observe
    def my_agent(query):
        return "Hello world"
"""

from agentpulse.observe import observe
from agentpulse.config import configure, get_config, reset_config
from agentpulse.transport import get_transport
from agentpulse.models import SpanData

def init(*args, **kwargs):
    """Universal initialization for AgentPulse SDK.
    Supports both lightweight @observe configuration (backend_url, api_key)
    and legacy AgentPulseClient singleton management (model, persist, etc.).
    """
    if "agent_id" in kwargs and "agent_name" not in kwargs:
        kwargs["agent_name"] = kwargs["agent_id"]
    if "version" in kwargs and "agent_version" not in kwargs:
        kwargs["agent_version"] = kwargs["version"]

    cfg_keys = {"backend_url", "api_key", "agent_name", "agent_version", "environment", "disabled", "timeout"}
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}
    if cfg_kwargs:
        configure(**cfg_kwargs)

    try:
        from src.sdk.client import init as legacy_init
        return legacy_init(*args, **kwargs)
    except Exception:
        return None


def flush(timeout: float = 5.0):
    """Wait for all pending telemetry to be transmitted."""
    get_transport().flush(timeout=timeout)


class AgentPulse:
    """AgentPulse entry point for global SDK configuration and lifecycle."""
    init = staticmethod(init)
    configure = staticmethod(configure)
    flush = staticmethod(flush)

    def __init__(self, *args, **kwargs):
        self.config = init(*args, **kwargs)
        cfg = get_config()
        self.api_key = cfg.api_key
        self.agent_id = cfg.agent_name
        self.agent_version = cfg.agent_version
        self.backend_url = cfg.backend_url


# Backward-compatible and advanced evaluation utilities
try:
    from src.sdk import (
        AgentPulseClient,
        get_client,
        trace,
        tool,
        mcp_tool,
        evaluate,
        CallableAgentAdapter,
    )
except ImportError:
    # Standalone mode outside framework root
    AgentPulseClient = None
    get_client = None
    trace = None
    tool = None
    mcp_tool = None
    evaluate = None
    CallableAgentAdapter = None

__version__ = "1.0.0"

__all__ = [
    "observe",
    "configure",
    "get_config",
    "reset_config",
    "flush",
    "SpanData",
    "AgentPulseClient",
    "AgentPulse",
    "init",
    "get_client",
    "trace",
    "tool",
    "mcp_tool",
    "evaluate",
    "CallableAgentAdapter",
    "__version__",
]
