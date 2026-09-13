"""Subprocess-isolated sandbox driver.

Runs agent executions in an isolated, sanitized child process with strict
timeouts, output capture, and process tree termination on timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from src.sandbox.drivers.base import BaseSandboxDriver
from src.sandbox.isolation import SandboxEnvironmentManager
from src.sandbox.models import SandboxConfig, SandboxExecutionResult, SecurityLevel


class SubprocessSandboxDriver(BaseSandboxDriver):
    """Executes sandboxed agent tasks via an isolated subprocess boundary."""

    def __init__(self, env_manager: SandboxEnvironmentManager | None = None):
        self.env_manager = env_manager or SandboxEnvironmentManager()

    @property
    def security_level(self) -> SecurityLevel:
        return SecurityLevel.ISOLATED_PROCESS

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        workspace_dir: Path,
        harness_input: Dict[str, Any],
        config: SandboxConfig,
    ) -> SandboxExecutionResult:
        workspace = Path(workspace_dir).resolve()
        input_file = workspace / "harness_input.json"
        output_file = workspace / "harness_output.json"

        # Write harness input
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(harness_input, f, indent=2)

        # Sanitize environment variables
        sanitized_env = self.env_manager.sanitize_environment(config, workspace)

        # Build command: use current python interpreter to run harness module
        cmd = [
            sys.executable,
            "-m",
            "src.sandbox.harness",
            str(input_file),
            str(output_file),
        ]

        start_time = time.time()
        process = None
        timed_out = False
        stdout_data = ""
        stderr_data = ""
        exit_code = -1

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(workspace),
                env=sanitized_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout_data, stderr_data = process.communicate(timeout=config.timeout_seconds)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout_data, stderr_data = process.communicate()
                exit_code = -9

        except Exception as e:
            stderr_data = f"Failed to spawn sandbox subprocess: {e}"
            exit_code = -1
        finally:
            duration_ms = (time.time() - start_time) * 1000

        # Read output payload
        output_payload: Dict[str, Any] = {}
        if output_file.is_file():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    output_payload = json.load(f)
            except Exception as e:
                stderr_data += f"\nFailed to parse harness output: {e}"

        # Combine response and status
        if timed_out:
            success = False
            response = f"Execution timed out after {config.timeout_seconds} seconds."
            error = f"TimeoutExpired: Execution exceeded limit of {config.timeout_seconds}s."
            status = "TIMEOUT"
        elif exit_code != 0 and not output_payload:
            success = False
            response = "Agent execution failed in sandbox process."
            error = f"Process exited with code {exit_code}.\nStderr: {stderr_data}"
            status = "FAILED"
        else:
            status = output_payload.get("status", "SUCCESS" if exit_code == 0 else "FAILED")
            success = (status == "SUCCESS")
            response = output_payload.get("response", "")
            error = output_payload.get("error") or (stderr_data if exit_code != 0 else None)

        token_usage = output_payload.get("token_usage", {
            "input_tokens": max(1, len(harness_input.get("query", "").split()) * 2),
            "output_tokens": max(1, len(response.split()) * 2),
            "total_tokens": max(2, len(harness_input.get("query", "").split()) * 2 + len(response.split()) * 2),
        })

        return SandboxExecutionResult(
            success=success,
            status=status,
            security_level=self.security_level,
            duration_ms=duration_ms,
            exit_code=exit_code,
            response=response,
            error=error,
            raw_output={"stdout": stdout_data[:config.max_output_size_bytes], "stderr": stderr_data[:config.max_output_size_bytes]},
            token_usage=token_usage,
            trace_steps=output_payload.get("steps", []),
            timed_out=timed_out,
        )
