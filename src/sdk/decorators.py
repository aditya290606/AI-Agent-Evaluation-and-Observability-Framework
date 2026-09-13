"""
Decorators for the AgentPulse SDK.

Provides:
  - @trace: Decorates agent functions to record full execution runs.
  - @tool: Decorates internal tool functions to capture inputs, outputs, and latency.
  - @mcp_tool: Decorates MCP tool functions with server metadata and execution tracking.
"""

from functools import wraps
import inspect
import time
from typing import Optional, Callable, Any, Dict

from src.sdk.client import get_client, _current_trace_var
from src.core.entities import Trace
from src.tracing.tracer import RunTrace


def trace(
    name: Optional[str] = None,
    agent_name: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
    persist: bool = True,
):
    """Decorator to instrument an external agent entry point.

    Usage:
        @agentpulse.trace(agent_name="SupportBot", version="v1.2")
        def run_agent(query: str) -> str:
            result = call_llm(query)
            return result
    """
    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            client = get_client()

            # Extract query and task_id intelligently from args/kwargs
            query_str = ""
            task_id = kwargs.get("task_id")

            # Check kwargs first
            if "query" in kwargs:
                query_str = str(kwargs["query"])
            elif "user_input" in kwargs:
                query_str = str(kwargs["user_input"])
            elif "prompt" in kwargs:
                query_str = str(kwargs["prompt"])
            elif "input_text" in kwargs:
                query_str = str(kwargs["input_text"])
            elif args:
                # If first arg looks like a task_id and second is query (common in benchmarks)
                if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str) and not task_id:
                    if len(args[0]) < 20 and ("task" in args[0].lower() or "t_" in args[0].lower() or "t0" in args[0].lower()):
                        task_id = args[0]
                        query_str = args[1]
                    else:
                        query_str = str(args[0])
                else:
                    query_str = str(args[0])

            # If inside an existing evaluation or trace context, invoke directly
            existing_trace = _current_trace_var.get()
            if existing_trace is not None:
                return fn(*args, **kwargs)

            # Start fresh trace context
            with client.trace(
                name=fn_name,
                query=query_str,
                task_id=task_id,
                agent_name=agent_name,
                agent_version=version,
                model=model,
                persist=persist,
            ) as t:
                result = fn(*args, **kwargs)

                # If the function returns a Trace or RunTrace directly, adopt its answer
                if isinstance(result, (RunTrace, Trace)):
                    t.final_answer = result.final_answer
                    # Merge any spans if not already logged
                    if not t.spans and result.spans:
                        t.spans = result.spans
                else:
                    t.finish(str(result) if result is not None else "")

                # Attach last trace reference for inspection
                wrapper._last_trace = t
                return result

        wrapper._is_agentpulse_traced = True
        return wrapper

    return decorator


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to instrument a tool function called by an agent.

    Usage:
        @agentpulse.tool(name="calculator")
        def calculate(expression: str) -> str:
            return str(eval(expression))
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            client = get_client()
            trace_obj = client.current_trace

            if not trace_obj:
                # No active trace; execute normally
                return fn(*args, **kwargs)

            # Record call arguments
            call_args: Dict[str, Any] = {}
            if kwargs:
                call_args.update(kwargs)
            if args:
                call_args["args"] = args

            input_repr = kwargs.get("query") or kwargs.get("input") or (args[0] if len(args) == 1 else call_args)

            start = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000

                client.log_tool_call(
                    tool_name=tool_name,
                    input_data=input_repr,
                    output_data=result,
                    latency_ms=elapsed_ms,
                    status="success",
                    is_mcp=False,
                )
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                client.log_tool_call(
                    tool_name=tool_name,
                    input_data=input_repr,
                    output_data=f"ERROR: {str(e)}",
                    latency_ms=elapsed_ms,
                    status="error",
                    error=str(e),
                    is_mcp=False,
                )
                raise

        wrapper._is_agentpulse_tool = True
        wrapper.tool_name = tool_name
        return wrapper

    return decorator


def mcp_tool(server: str, name: Optional[str] = None):
    """Decorator to instrument an MCP (Model Context Protocol) tool call.

    Usage:
        @agentpulse.mcp_tool(server="Knowledge Base", name="search_documents")
        def search_kb(query: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            client = get_client()
            trace_obj = client.current_trace

            if not trace_obj:
                return fn(*args, **kwargs)

            call_args: Dict[str, Any] = {}
            if kwargs:
                call_args.update(kwargs)
            if args:
                call_args["args"] = args

            input_repr = kwargs.get("query") or kwargs.get("input") or (args[0] if len(args) == 1 else call_args)

            start = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000

                client.log_mcp_call(
                    server_name=server,
                    tool_name=tool_name,
                    input_data=input_repr,
                    output_data=result,
                    latency_ms=elapsed_ms,
                    status="success",
                )
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                client.log_mcp_call(
                    server_name=server,
                    tool_name=tool_name,
                    input_data=input_repr,
                    output_data=f"ERROR: {str(e)}",
                    latency_ms=elapsed_ms,
                    status="error",
                    error=str(e),
                )
                raise

        wrapper._is_agentpulse_mcp = True
        wrapper.mcp_server = server
        wrapper.tool_name = tool_name
        return wrapper

    return decorator
