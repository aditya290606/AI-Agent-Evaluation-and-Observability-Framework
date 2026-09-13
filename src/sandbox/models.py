"""Data models and enumerations for the Secure Sandbox Execution Architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SecurityLevel(str, Enum):
    """Execution boundary isolation grade."""
    INSECURE_DIRECT = "INSECURE_DIRECT"           # Executed in main application memory (Forbidden for uploads)
    ISOLATED_PROCESS = "ISOLATED_PROCESS"         # Ephemeral subprocess with memory/timeout limits & sanitized env
    CONTAINER_SANDBOX = "CONTAINER_SANDBOX"       # Full OCI/Docker container boundary with cgroups & dropped caps


class SecurityRiskLevel(str, Enum):
    """Overall project risk assessment level."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityCategory(str, Enum):
    """Category of suspicious or restricted code behavior."""
    DANGEROUS_EVAL = "DANGEROUS_EVAL"             # eval, exec, compile
    CODE_EXECUTION = "CODE_EXECUTION"             # eval, exec, compile
    SYSTEM_COMMAND = "SYSTEM_COMMAND"             # os.system, os.popen
    PROCESS_SPAWN = "PROCESS_SPAWN"               # subprocess.Popen, multiprocessing
    HOST_FILE_ACCESS = "HOST_FILE_ACCESS"         # targeting /etc/passwd, C:\Windows, .env, SSH keys
    FILESYSTEM_ACCESS = "FILESYSTEM_ACCESS"       # file deletions, directory access
    NETWORK_ACCESS = "NETWORK_ACCESS"             # socket, urllib, raw connections
    CREDENTIAL_ACCESS = "CREDENTIAL_ACCESS"       # scanning host env vars or key files
    DANGEROUS_IMPORT = "DANGEROUS_IMPORT"         # ctypes, win32api, pty
    SUSPICIOUS_OPERATIONS = "SUSPICIOUS_OPERATIONS" # generic suspicious syntax / parse errors


class NetworkPolicy(str, Enum):
    """Sandbox network isolation policy."""
    OFFLINE = "OFFLINE"                           # Completely blocked network access
    RESTRICTED = "RESTRICTED"                     # Allow only specified LLM provider endpoints
    ALLOW_ALL = "ALLOW_ALL"                       # Full internet access (Standard LLM / web agent mode)


@dataclass
class SecurityAlert:
    """A detected suspicious or hazardous code construct from static inspection."""
    file_path: str = ""
    line_number: int = 0
    risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW
    category: SecurityCategory = SecurityCategory.CODE_EXECUTION
    message: str = ""
    rule_id: str = ""
    code_snippet: str = ""
    severity: Optional[SecurityRiskLevel] = None

    def __post_init__(self):
        if self.severity is not None:
            self.risk_level = self.severity
        else:
            self.severity = self.risk_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "risk_level": self.risk_level.value,
            "severity": self.risk_level.value,
            "category": self.category.value,
            "message": self.message,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ProjectManifest:
    """Metadata extracted during static project analysis."""
    entry_point: str = "agent.py"
    entry_symbol: str = "run"
    framework: str = "custom"                     # langgraph, crewai, autogen, llamaindex, langchain, etc.
    dependencies: List[str] = field(default_factory=list)
    required_env_vars: List[str] = field(default_factory=list)
    description: str = ""
    total_files: int = 0
    python_files: List[str] = field(default_factory=list)
    total_size_bytes: int = 0
    project_name: str = "uploaded_agent"

    @property
    def entry_point_file(self) -> str:
        return self.entry_point

    @entry_point_file.setter
    def entry_point_file(self, val: str):
        self.entry_point = val

    @property
    def callable_name(self) -> str:
        return self.entry_symbol

    @callable_name.setter
    def callable_name(self, val: str):
        self.entry_symbol = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "entry_point_file": self.entry_point,
            "entry_symbol": self.entry_symbol,
            "callable_name": self.entry_symbol,
            "framework": self.framework,
            "dependencies": self.dependencies,
            "required_env_vars": self.required_env_vars,
            "description": self.description,
            "total_files": self.total_files,
            "project_name": self.project_name,
        }


@dataclass
class StaticInspectionResult:
    """Comprehensive static inspection and security audit results."""
    manifest: ProjectManifest
    alerts: List[SecurityAlert] = field(default_factory=list)
    overall_risk: SecurityRiskLevel = SecurityRiskLevel.LOW
    risk_score: float = 0.0                       # 0.0 (Safe) to 100.0 (Critical Threat)
    is_safe_to_execute: bool = True
    rejection_reason: Optional[str] = None
    lines_of_code: int = 0
    file_count: int = 0
    summary_findings: List[str] = field(default_factory=list)

    @property
    def security_alerts(self) -> List[SecurityAlert]:
        return self.alerts

    @property
    def risk_level(self) -> SecurityRiskLevel:
        return self.overall_risk

    @property
    def is_runnable(self) -> bool:
        return self.is_safe_to_execute

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "alerts": [a.to_dict() for a in self.alerts],
            "security_alerts": [a.to_dict() for a in self.alerts],
            "overall_risk": self.overall_risk.value,
            "risk_level": self.overall_risk.value,
            "risk_score": round(self.risk_score, 1),
            "is_safe_to_execute": self.is_safe_to_execute,
            "is_runnable": self.is_safe_to_execute,
            "rejection_reason": self.rejection_reason,
            "lines_of_code": self.lines_of_code,
            "file_count": self.file_count,
            "summary_findings": self.summary_findings,
        }


@dataclass
class SandboxConfig:
    """Resource limits and isolation parameters for sandbox execution."""
    timeout_seconds: float = 30.0                 # Max execution time per test task
    max_memory_mb: int = 512                      # Memory limit in Megabytes
    max_cpu_percent: float = 100.0                # CPU usage cap
    max_output_size_bytes: int = 500_000          # Output buffer limit
    max_output_kb: int = 500
    network_policy: NetworkPolicy = NetworkPolicy.ALLOW_ALL
    env_allowlist: List[str] = field(default_factory=list)
    custom_env_vars: Dict[str, str] = field(default_factory=dict)
    driver_type: str = "subprocess"               # "subprocess" | "docker"
    cleanup_on_finish: bool = True                # Destroy ephemeral directory immediately
    custom_python_executable: Optional[str] = None

    @property
    def allowed_env_vars(self) -> Dict[str, str]:
        return self.custom_env_vars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_output_size_bytes": self.max_output_size_bytes,
            "network_policy": self.network_policy.value,
            "driver_type": self.driver_type,
            "cleanup_on_finish": self.cleanup_on_finish,
            "custom_env_vars_count": len(self.custom_env_vars),
        }


@dataclass
class SandboxExecutionResult:
    """Outcome of an isolated sandbox execution task."""
    success: bool
    status: str = "SUCCESS"
    security_level: SecurityLevel = SecurityLevel.ISOLATED_PROCESS
    duration_ms: float = 0.0
    exit_code: Optional[int] = 0
    response: str = ""
    error: Optional[str] = None
    raw_output: Dict[str, str] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    trace_steps: List[Dict[str, Any]] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    timed_out: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def output(self) -> str:
        return self.response

    @property
    def latency_ms(self) -> float:
        return self.duration_ms

    @property
    def was_timeout(self) -> bool:
        return self.timed_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "security_level": self.security_level.value,
            "duration_ms": round(self.duration_ms, 2),
            "latency_ms": round(self.duration_ms, 2),
            "exit_code": self.exit_code,
            "response": self.response,
            "output": self.response,
            "error": self.error,
            "token_usage": self.token_usage,
            "trace_steps_count": len(self.trace_steps),
            "timed_out": self.timed_out,
            "was_timeout": self.timed_out,
        }
