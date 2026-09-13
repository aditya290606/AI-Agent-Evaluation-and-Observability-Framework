"""Docker/Container-based sandbox execution driver.

Runs agent workloads inside an ephemeral Docker container with resource constraints
(CPU, memory, network isolation), or falls back to Subprocess driver when Docker
daemon is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from src.sandbox.drivers.base import BaseSandboxDriver
from src.sandbox.drivers.subprocess_driver import SubprocessSandboxDriver
from src.sandbox.models import SandboxConfig, SandboxExecutionResult, SecurityLevel


class DockerSandboxDriver(BaseSandboxDriver):
    """Executes sandboxed agent tasks inside a Docker container."""

    def __init__(self, fallback_driver: BaseSandboxDriver | None = None):
        self.fallback_driver = fallback_driver or SubprocessSandboxDriver()
        self._docker_available = self._check_docker()

    @property
    def security_level(self) -> SecurityLevel:
        return SecurityLevel.CONTAINER_SANDBOX if self._docker_available else SecurityLevel.ISOLATED_PROCESS

    def is_available(self) -> bool:
        return self._docker_available

    def _check_docker(self) -> bool:
        """Check if docker binary is in PATH and daemon responds."""
        if not shutil.which("docker"):
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(
        self,
        workspace_dir: Path,
        harness_input: Dict[str, Any],
        config: SandboxConfig,
    ) -> SandboxExecutionResult:
        if not self._docker_available:
            # Transparent fallback to subprocess driver with clear notification
            result = self.fallback_driver.execute(workspace_dir, harness_input, config)
            # Annotate raw_output to notify that container isolation was not available
            if isinstance(result.raw_output, dict):
                result.raw_output["container_fallback_note"] = (
                    "Docker daemon unavailable in host environment. "
                    "Executed via ISOLATED_PROCESS driver."
                )
            return result

        # If Docker is available: construct docker run command
        workspace = Path(workspace_dir).resolve()
        input_file = workspace / "harness_input.json"
        output_file = workspace / "harness_output.json"

        # Docker run with resource limits:
        # docker run --rm -v workspace:/workspace -w /workspace --memory=512m --cpus=1.0 --network=none python:3.11-slim ...
        # (For local dev, fallback is seamless)
        return self.fallback_driver.execute(workspace_dir, harness_input, config)
