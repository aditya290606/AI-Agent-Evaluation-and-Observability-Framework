"""Secure sandbox execution architecture for evaluating uploaded AI-agent projects."""

from src.sandbox.adapter import SandboxedAgentAdapter
from src.sandbox.drivers.base import BaseSandboxDriver
from src.sandbox.drivers.docker_driver import DockerSandboxDriver
from src.sandbox.drivers.subprocess_driver import SubprocessSandboxDriver
from src.sandbox.inspector import ASTSecurityVisitor, ProjectInspector
from src.sandbox.isolation import SandboxEnvironmentManager
from src.sandbox.manager import SandboxManager
from src.sandbox.models import (
    ProjectManifest,
    SandboxConfig,
    SandboxExecutionResult,
    SecurityAlert,
    SecurityCategory,
    SecurityLevel,
    SecurityRiskLevel,
    StaticInspectionResult,
)
from src.sandbox.validator import ArchiveValidationError, ArchiveValidator

__all__ = [
    "SecurityLevel",
    "SecurityRiskLevel",
    "SecurityCategory",
    "SecurityAlert",
    "ProjectManifest",
    "StaticInspectionResult",
    "SandboxConfig",
    "SandboxExecutionResult",
    "ArchiveValidator",
    "ArchiveValidationError",
    "ASTSecurityVisitor",
    "ProjectInspector",
    "SandboxEnvironmentManager",
    "BaseSandboxDriver",
    "SubprocessSandboxDriver",
    "DockerSandboxDriver",
    "SandboxManager",
    "SandboxedAgentAdapter",
]
