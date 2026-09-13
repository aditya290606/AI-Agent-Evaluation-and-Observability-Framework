"""Abstract base class for sandbox execution drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from src.sandbox.models import SandboxConfig, SandboxExecutionResult, SecurityLevel


class BaseSandboxDriver(ABC):
    """Abstract interface for sandbox execution drivers."""

    @property
    @abstractmethod
    def security_level(self) -> SecurityLevel:
        """Declared security level provided by this driver."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the driver runtime is available in current environment."""
        pass

    @abstractmethod
    def execute(
        self,
        workspace_dir: Path,
        harness_input: Dict[str, Any],
        config: SandboxConfig,
    ) -> SandboxExecutionResult:
        """Execute sandboxed workload and return standardized execution result."""
        pass
