"""Standalone execution harness script for sandboxed agent invocations.

This script is executed INSIDE the isolated sandbox process or container.
It receives input via an input JSON file, invokes the target agent module,
captures all traces, outputs, metrics, and errors, and writes an output JSON.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


def run_sandboxed_target(input_file: str, output_file: str) -> None:
    start_time = time.time()
    trace_steps: List[Dict[str, Any]] = []
    response_text = ""
    error_message: str | None = None
    status = "SUCCESS"
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        project_dir = payload.get("project_dir", ".")
        entry_point = payload.get("entry_point", "agent.py")
        entry_symbol = payload.get("entry_symbol", "run")
        task_id = payload.get("task_id", "sandbox_task")
        query = payload.get("query", "")
        max_output_size = payload.get("max_output_size_bytes", 500_000)

        # Ensure project directory and entry file directory are in sys.path
        abs_project_dir = os.path.abspath(project_dir)
        if abs_project_dir not in sys.path:
            sys.path.insert(0, abs_project_dir)

        # Resolve entry file
        abs_entry_file = os.path.abspath(os.path.join(abs_project_dir, entry_point))
        if not os.path.isfile(abs_entry_file):
            # Try searching for filename inside project
            matched = list(Path(abs_project_dir).glob(f"**/{entry_point}"))
            if not matched and not entry_point.endswith(".py"):
                matched = list(Path(abs_project_dir).glob(f"**/{entry_point}.py"))
            if matched:
                abs_entry_file = str(matched[0].resolve())
            else:
                # Fallback to any python file in directory
                all_py = list(Path(abs_project_dir).glob("**/*.py"))
                if all_py:
                    abs_entry_file = str(all_py[0].resolve())
                else:
                    raise FileNotFoundError(f"Entry point file '{abs_entry_file}' does not exist.")

        entry_dir = os.path.dirname(abs_entry_file)
        if entry_dir not in sys.path:
            sys.path.insert(0, entry_dir)

        # Dynamically load the module
        module_name = f"sandboxed_module_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, abs_entry_file)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from '{abs_entry_file}'")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find target callable / class
        target = getattr(module, entry_symbol, None)
        if target is None:
            # Fallback to search for common entry functions or any function/class in module
            for fallback in ["run", "execute", "invoke", "call", "agent", "run_agent_mock", "run_agent", "main"]:
                if hasattr(module, fallback):
                    target = getattr(module, fallback)
                    break

        if target is None:
            # Pick first callable or class defined in module
            for attr_name in dir(module):
                if not attr_name.startswith("_"):
                    obj = getattr(module, attr_name)
                    if (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, "__module__", "") == module_name:
                        target = obj
                        break

        if target is None:
            raise AttributeError(f"Could not find callable '{entry_symbol}' or standard fallback in '{os.path.basename(abs_entry_file)}'")

        # Instantiate class if class is provided
        if inspect.isclass(target):
            try:
                instance = target()
                # Search for run/execute/invoke/call method
                found_method = False
                for m_name in ["run", "execute", "invoke", "call", "step", "__call__"]:
                    if hasattr(instance, m_name) and callable(getattr(instance, m_name)):
                        target = getattr(instance, m_name)
                        found_method = True
                        break
                if not found_method:
                    raise AttributeError(f"Class '{target.__name__}' does not have a run(), execute(), or invoke() method.")
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate agent class '{entry_symbol}': {e}")

        if not callable(target):
            raise TypeError(f"Target '{entry_symbol}' in '{os.path.basename(abs_entry_file)}' is not callable.")

        # Execute agent with task query
        step_start = time.time()
        sig = inspect.signature(target)
        kwargs: Dict[str, Any] = {}

        params = list(sig.parameters.values())
        if len(params) == 0:
            raw_result = target()
        elif len(params) == 1:
            raw_result = target(query)
        else:
            param_names = [p.name for p in params]
            if "task_id" in param_names and "query" in param_names:
                kwargs["task_id"] = task_id
                kwargs["query"] = query
                raw_result = target(**kwargs)
            elif "query" in param_names:
                kwargs["query"] = query
                raw_result = target(**kwargs)
            elif "prompt" in param_names:
                kwargs["prompt"] = query
                raw_result = target(**kwargs)
            elif "input" in param_names:
                kwargs["input"] = query
                raw_result = target(**kwargs)
            elif "user_input" in param_names:
                kwargs["user_input"] = query
                raw_result = target(**kwargs)
            else:
                # Fill first arg as query, rest as defaults or mock
                args_list = [query]
                for p in params[1:]:
                    if p.default != inspect.Parameter.empty:
                        args_list.append(p.default)
                    else:
                        args_list.append(None)
                raw_result = target(*args_list)

        # Handle asynchronous / coroutine result if returned
        if inspect.iscoroutine(raw_result):
            import asyncio
            raw_result = asyncio.run(raw_result)

        step_duration = (time.time() - step_start) * 1000

        # Parse agent output
        if hasattr(raw_result, "final_answer"):
            response_text = str(raw_result.final_answer)
            if hasattr(raw_result, "spans"):
                for idx, sp in enumerate(raw_result.spans):
                    trace_steps.append({
                        "step_index": getattr(sp, "step_index", idx + 1),
                        "step_type": getattr(sp, "step_type", "agent"),
                        "tool_name": getattr(sp, "tool_name", None),
                        "content": str(getattr(sp, "input_data", "")),
                        "output": str(getattr(sp, "output_data", "")),
                        "latency_ms": getattr(sp, "latency_ms", 0.0),
                        "status": getattr(sp, "status", "success"),
                    })
        elif isinstance(raw_result, dict):
            response_text = str(raw_result.get("response", raw_result.get("output", raw_result.get("answer", ""))))
            if not response_text:
                response_text = json.dumps(raw_result)

            if "steps" in raw_result and isinstance(raw_result["steps"], list):
                trace_steps = raw_result["steps"]
            if "usage" in raw_result and isinstance(raw_result["usage"], dict):
                token_usage.update(raw_result["usage"])
        elif isinstance(raw_result, str):
            response_text = raw_result
        else:
            response_text = str(raw_result)

        # Create standard step if none were recorded
        if not trace_steps:
            trace_steps.append({
                "step_index": 1,
                "step_type": "THOUGHT_AND_ACTION",
                "content": f"Processed query: {query}",
                "latency_ms": step_duration,
                "status": "SUCCESS",
            })

    except Exception as e:
        status = "FAILED"
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        response_text = f"Execution failed: {e}"

    duration_ms = (time.time() - start_time) * 1000

    # Secret Redaction and Safety Guardrails
    try:
        from src.security.redactor import SecretRedactor
        response_text = SecretRedactor.redact_text(response_text)
        if error_message:
            error_message = SecretRedactor.redact_text(error_message)
        trace_steps = SecretRedactor.redact_data(trace_steps)
    except Exception:
        pass

    # Truncate response if exceeding limit
    if len(response_text) > max_output_size:
        response_text = response_text[:max_output_size] + f"\n... [TRUNCATED: Output exceeded {max_output_size} bytes limit]"

    # Estimate tokens if none provided
    if token_usage["total_tokens"] == 0:
        in_toks = max(1, len(query.split()) * 2)
        out_toks = max(1, len(response_text.split()) * 2)
        token_usage = {
            "input_tokens": in_toks,
            "output_tokens": out_toks,
            "total_tokens": in_toks + out_toks,
        }

    output_payload = {
        "status": status,
        "response": response_text,
        "error": error_message,
        "duration_ms": duration_ms,
        "token_usage": token_usage,
        "steps": trace_steps,
    }

    # Ensure output directory exists and write result
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.sandbox.harness <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    run_sandboxed_target(input_path, output_path)
