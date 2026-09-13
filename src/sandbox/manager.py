"""High-level Sandbox Manager coordinating validation, static inspection, provisioning, execution, and cleanup."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from src.sandbox.drivers.base import BaseSandboxDriver
from src.sandbox.drivers.subprocess_driver import SubprocessSandboxDriver
from src.sandbox.inspector import ProjectInspector
from src.sandbox.isolation import SandboxEnvironmentManager
from src.sandbox.models import (
    ProjectManifest,
    SandboxConfig,
    SandboxExecutionResult,
    StaticInspectionResult,
)
from src.sandbox.validator import ArchiveValidator


class SandboxManager:
    """Orchestrates secure project validation, static AST inspection, and sandboxed execution."""

    def __init__(
        self,
        driver: Optional[BaseSandboxDriver] = None,
        env_manager: Optional[SandboxEnvironmentManager] = None,
        inspector: Optional[ProjectInspector] = None,
        validator: Optional[ArchiveValidator] = None,
    ):
        self.env_manager = env_manager or SandboxEnvironmentManager()
        self.driver = driver or SubprocessSandboxDriver(self.env_manager)
        self.inspector = inspector or ProjectInspector()
        self.validator = validator or ArchiveValidator()

    def validate_and_extract(self, archive_path: Path, target_dir: Optional[Path] = None) -> Path:
        """Validate an uploaded archive (zip/tar) and safely extract it."""
        if target_dir is None:
            extract_base = Path(tempfile.gettempdir()) / "agent_eval_uploads" / f"proj_{uuid.uuid4().hex[:8]}"
            extract_base.mkdir(parents=True, exist_ok=True)
            target_dir = extract_base

        return self.validator.validate_and_extract(archive_path, target_dir)

    def inspect_project(self, project_dir: Path) -> StaticInspectionResult:
        """Perform static AST analysis and dependency extraction without running user code."""
        return self.inspector.inspect_directory(project_dir)

    def execute_task(
        self,
        project_dir: Path,
        task_id: str,
        query: str,
        entry_point: str = "agent.py",
        entry_symbol: str = "run",
        config: Optional[SandboxConfig] = None,
    ) -> SandboxExecutionResult:
        """Provision ephemeral workspace, run task in isolated process, capture results, and destroy workspace."""
        config = config or SandboxConfig()

        # Create ephemeral workspace
        workspace = self.env_manager.create_ephemeral_workspace(project_dir)

        try:
            harness_input = {
                "project_dir": str(workspace),
                "entry_point": entry_point,
                "entry_symbol": entry_symbol,
                "task_id": task_id,
                "query": query,
                "max_output_size_bytes": config.max_output_size_bytes,
            }

            result = self.driver.execute(
                workspace_dir=workspace,
                harness_input=harness_input,
                config=config,
            )
            return result
        finally:
            # Guaranteed workspace destruction
            self.env_manager.cleanup_workspace(workspace)
