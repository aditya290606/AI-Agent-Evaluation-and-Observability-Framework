"""
The @observe decorator for AgentPulse SDK.

Automatically captures:
  - trace_id
  - span_id
  - function name
  - input
  - output
  - start time
  - end time
  - latency
  - success/failure status
  - exception information

Supports:
  - Synchronous and asynchronous functions
  - Nested observed functions
  - Zero-overhead safe error handling (user code never fails due to telemetry)
"""

from functools import wraps
import inspect
import time
import traceback
import uuid
from typing import Optional, Callable, Any, Dict

from agentpulse.context import (
    get_current_trace_id,
    get_current_span_id,
    get_current_agent_id,
    get_current_agent_version,
    get_current_run_id,
    set_context,
    reset_context,
)
from agentpulse.models import SpanData
from agentpulse.transport import get_transport
from agentpulse.config import get_config


def _gen_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def _format_input(args: tuple, kwargs: dict) -> Any:
    """Extract and format function input parameters."""
    if not args and not kwargs:
        return ""
    if len(args) == 1 and not kwargs:
        return args[0]
    if not args and len(kwargs) == 1:
        return next(iter(kwargs.values()))

    payload = {}
    if args:
        payload["args"] = args
    if kwargs:
        payload.update(kwargs)
    return payload


def observe(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    span_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_version: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Decorator to observe an agent or tool function with automatic telemetry.

    Can be used as:
        @observe
        def my_agent(query):
            return ...

        @observe(name="CustomAgent", agent_id="my_agent", agent_version="1.0.0")
        async def my_async_agent(query):
            return ...
    """
    def decorator(target_fn: Callable) -> Callable:
        fn_name = name or getattr(target_fn, "__qualname__", None) or getattr(target_fn, "__name__", "observed_function")
        stype = span_type or "function"
        config = get_config()
        meta = dict(metadata or {})
        aid = agent_id or meta.get("agent_id") or config.agent_name
        aver = agent_version or meta.get("agent_version") or config.agent_version
        if "agent_name" not in meta:
            meta["agent_name"] = aid
        if "agent_id" not in meta:
            meta["agent_id"] = aid
        if "agent_version" not in meta:
            meta["agent_version"] = aver
        if run_id and "run_id" not in meta:
            meta["run_id"] = run_id
        if "environment" not in meta:
            meta["environment"] = config.environment

        if inspect.iscoroutinefunction(target_fn):
            @wraps(target_fn)
            async def async_wrapper(*args, **kwargs) -> Any:
                trace_id = get_current_trace_id() or _gen_id("trc")
                parent_span_id = get_current_span_id()
                span_id = _gen_id("spn")

                current_aid = agent_id or meta.get("agent_id") or get_current_agent_id() or config.agent_name
                current_aver = agent_version or meta.get("agent_version") or get_current_agent_version() or config.agent_version
                current_rid = run_id or meta.get("run_id") or get_current_run_id() or trace_id

                tokens = set_context(trace_id, span_id, current_aid, current_aver, current_rid)
                start_time = time.time()
                status = "success"
                output = None
                exc_info = None

                try:
                    output = await target_fn(*args, **kwargs)
                    return output
                except Exception as e:
                    status = "error"
                    exc_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    raise
                finally:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000.0

                    try:
                        span = SpanData(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=parent_span_id,
                            function_name=fn_name,
                            input=_format_input(args, kwargs),
                            output=output if exc_info is None else f"ERROR: {exc_info['message']}",
                            start_time=start_time,
                            end_time=end_time,
                            latency=latency_ms,
                            status=status,
                            exception_info=exc_info,
                            span_type=stype,
                            metadata=meta,
                            agent_id=current_aid,
                            agent_version=current_aver,
                            run_id=current_rid,
                        )
                        get_transport().send_span(span)
                        async_wrapper._last_span = span
                    except Exception:
                        pass
                    finally:
                        reset_context(tokens)

            async_wrapper._is_agentpulse_observed = True
            return async_wrapper
        else:
            @wraps(target_fn)
            def sync_wrapper(*args, **kwargs) -> Any:
                trace_id = get_current_trace_id() or _gen_id("trc")
                parent_span_id = get_current_span_id()
                span_id = _gen_id("spn")

                current_aid = agent_id or meta.get("agent_id") or get_current_agent_id() or config.agent_name
                current_aver = agent_version or meta.get("agent_version") or get_current_agent_version() or config.agent_version
                current_rid = run_id or meta.get("run_id") or get_current_run_id() or trace_id

                tokens = set_context(trace_id, span_id, current_aid, current_aver, current_rid)
                start_time = time.time()
                status = "success"
                output = None
                exc_info = None

                try:
                    output = target_fn(*args, **kwargs)
                    return output
                except Exception as e:
                    status = "error"
                    exc_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    raise
                finally:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000.0

                    try:
                        span = SpanData(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=parent_span_id,
                            function_name=fn_name,
                            input=_format_input(args, kwargs),
                            output=output if exc_info is None else f"ERROR: {exc_info['message']}",
                            start_time=start_time,
                            end_time=end_time,
                            latency=latency_ms,
                            status=status,
                            exception_info=exc_info,
                            span_type=stype,
                            metadata=meta,
                            agent_id=current_aid,
                            agent_version=current_aver,
                            run_id=current_rid,
                        )
                        get_transport().send_span(span)
                        sync_wrapper._last_span = span
                    except Exception:
                        pass
                    finally:
                        reset_context(tokens)

            sync_wrapper._is_agentpulse_observed = True
            return sync_wrapper

    if fn is not None:
        # Invoked as @observe without parentheses
        return decorator(fn)
    return decorator
