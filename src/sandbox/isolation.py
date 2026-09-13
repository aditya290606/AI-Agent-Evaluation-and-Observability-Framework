"""Ephemeral sandbox workspace isolation and environment sanitizer.

Manages isolated temporary workspaces, ensures complete cleanup, and strips
all host credentials, API keys, and sensitive environment variables from
the execution process boundary.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Set

from src.sandbox.models import SandboxConfig


class SandboxEnvironmentManager:
    """Manages ephemeral directories and sanitized environment variables for sandbox execution."""

    # System environment variables necessary for Python runtime execution
    STANDARD_SYSTEM_ENV_ALLOWLIST: Set[str] = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "COMSPEC",
        "WINDIR",
        "TMP",
        "TEMP",
        "LOCALAPPDATA",
        "APPDATA",
        "HOMEPATH",
        "USERPROFILE",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUNBUFFERED",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
    }

    # Sensitive patterns that must NEVER be passed from host to sandbox
    FORBIDDEN_ENV_PATTERNS = [
        "KEY",
        "SECRET",
        "TOKEN",
        "PASS",
        "CREDENTIAL",
        "AUTH",
        "AWS_",
        "AZURE_",
        "GCP_",
        "GOOGLE_",
        "OPENAI_",
        "ANTHROPIC_",
        "DATABASE_URL",
        "DSN",
    ]

    def __init__(self, base_sandbox_dir: Optional[Path] = None):
        self.base_sandbox_dir = (
            base_sandbox_dir or Path(tempfile.gettempdir()) / "agent_eval_sandboxes"
        ).resolve()
        self.base_sandbox_dir.mkdir(parents=True, exist_ok=True)

    def create_ephemeral_workspace(self, source_dir: Path) -> Path:
        """Create an ephemeral workspace directory and copy project contents into it."""
        session_id = f"sandbox_{uuid.uuid4().hex[:12]}"
        workspace_dir = self.base_sandbox_dir / session_id
        workspace_dir.mkdir(parents=True, exist_ok=False)

        # Copy source contents
        source_dir = Path(source_dir).resolve()
        for item in source_dir.iterdir():
            target = workspace_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        return workspace_dir

    def sanitize_environment(
        self,
        config: SandboxConfig,
        workspace_dir: Path,
    ) -> Dict[str, str]:
        """Construct a strictly isolated environment dictionary without host secrets."""
        sanitized_env: Dict[str, str] = {}

        # 1. Pull ONLY explicitly allowed system variables from host
        allowlist = set(config.env_allowlist).union(self.STANDARD_SYSTEM_ENV_ALLOWLIST)
        for key in allowlist:
            if key in os.environ:
                # Check that key is not accidentally carrying a sensitive pattern unless explicitly allowed in config
                is_sensitive_pattern = any(p in key.upper() for p in self.FORBIDDEN_ENV_PATTERNS)
                if not is_sensitive_pattern or key in config.env_allowlist:
                    sanitized_env[key] = os.environ[key]

        # 2. Add sandboxed runtime defaults
        sanitized_env["PYTHONUNBUFFERED"] = "1"
        sanitized_env["PYTHONDONTWRITEBYTECODE"] = "1"
        sanitized_env["AGENT_EVAL_SANDBOX"] = "1"
        sanitized_env["AGENT_WORKSPACE"] = str(workspace_dir)

        # Set PYTHONPATH to include workspace and agent-eval-framework root
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [str(workspace_dir), repo_root]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        sanitized_env["PYTHONPATH"] = os.pathsep.join(paths)

        # 3. Inject ONLY user-specified explicitly configured environment variables
        for k, v in config.custom_env_vars.items():
            sanitized_env[k] = str(v)

        return sanitized_env

    def cleanup_workspace(self, workspace_dir: Path) -> bool:
        """Safely destroy the ephemeral workspace directory."""
        try:
            workspace = Path(workspace_dir).resolve()
            # Safety check: ensure workspace is indeed a child of base_sandbox_dir
            if self.base_sandbox_dir in workspace.parents or workspace == self.base_sandbox_dir:
                if workspace != self.base_sandbox_dir:
                    shutil.rmtree(workspace, ignore_errors=True)
                    return True
        except Exception:
            pass
        return False
