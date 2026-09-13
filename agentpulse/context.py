"""
Execution context tracking for AgentPulse SDK.

Uses contextvars for asynchronous and multi-threaded isolation.
Tracks active trace_id, parent_span_id, and nesting depth.
"""

from contextvars import ContextVar
from typing import Optional, Tuple, Any


_current_trace_id: ContextVar[Optional[str]] = ContextVar("agentpulse_current_trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("agentpulse_current_span_id", default=None)
_nesting_depth: ContextVar[int] = ContextVar("agentpulse_nesting_depth", default=0)
_current_agent_id: ContextVar[Optional[str]] = ContextVar("agentpulse_current_agent_id", default=None)
_current_agent_version: ContextVar[Optional[str]] = ContextVar("agentpulse_current_agent_version", default=None)
_current_run_id: ContextVar[Optional[str]] = ContextVar("agentpulse_current_run_id", default=None)


def get_current_trace_id() -> Optional[str]:
    """Return the active trace ID in the current execution context."""
    return _current_trace_id.get()


def get_current_span_id() -> Optional[str]:
    """Return the active span ID in the current execution context."""
    return _current_span_id.get()


def get_nesting_depth() -> int:
    """Return current nesting depth of observed functions."""
    return _nesting_depth.get()


def get_current_agent_id() -> Optional[str]:
    """Return the active agent_id in current context."""
    return _current_agent_id.get()


def get_current_agent_version() -> Optional[str]:
    """Return the active agent_version in current context."""
    return _current_agent_version.get()


def get_current_run_id() -> Optional[str]:
    """Return the active run_id in current context."""
    return _current_run_id.get()


def set_context(
    trace_id: str,
    span_id: str,
    agent_id: Optional[str] = None,
    agent_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Set current trace and span context, returning tokens for reset."""
    t_token = _current_trace_id.set(trace_id)
    s_token = _current_span_id.set(span_id)
    d_token = _nesting_depth.set(_nesting_depth.get() + 1)
    aid = agent_id or _current_agent_id.get()
    aver = agent_version or _current_agent_version.get()
    rid = run_id or _current_run_id.get()
    aid_token = _current_agent_id.set(aid)
    aver_token = _current_agent_version.set(aver)
    rid_token = _current_run_id.set(rid)
    return t_token, s_token, d_token, aid_token, aver_token, rid_token


def reset_context(tokens: Tuple[Any, Any, Any, Any, Any, Any]) -> None:
    """Reset context to previous state using tokens."""
    t_token, s_token, d_token, aid_token, aver_token, rid_token = tokens
    _current_trace_id.reset(t_token)
    _current_span_id.reset(s_token)
    _nesting_depth.reset(d_token)
    _current_agent_id.reset(aid_token)
    _current_agent_version.reset(aver_token)
    _current_run_id.reset(rid_token)

