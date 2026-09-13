"""Comprehensive test suite for the Secure Sandbox Execution Architecture."""

import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

from src.core.entities import TestCase
from src.sandbox.adapter import SandboxedAgentAdapter
from src.sandbox.drivers.docker_driver import DockerSandboxDriver
from src.sandbox.drivers.subprocess_driver import SubprocessSandboxDriver
from src.sandbox.inspector import ASTSecurityVisitor, ProjectInspector
from src.sandbox.isolation import SandboxEnvironmentManager
from src.sandbox.manager import SandboxManager
from src.sandbox.models import (
    SandboxConfig,
    SecurityAlert,
    SecurityCategory,
    SecurityLevel,
    SecurityRiskLevel,
)
from src.sandbox.validator import ArchiveValidationError, ArchiveValidator


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_sandbox_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


class TestArchiveValidator:
    """Tests for secure archive validation and ZipSlip prevention."""

    def test_valid_zip_extraction(self, temp_dir):
        validator = ArchiveValidator()
        zip_path = temp_dir / "valid_agent.zip"
        target_dir = temp_dir / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("agent.py", "def run(q):\n    return 'Hello ' + q\n")
            zf.writestr("requirements.txt", "requests==2.31.0\n")

        extracted = validator.validate_and_extract(zip_path, target_dir)
        assert extracted == target_dir
        assert (target_dir / "agent.py").is_file()
        assert (target_dir / "requirements.txt").is_file()

    def test_zipslip_traversal_rejection(self, temp_dir):
        validator = ArchiveValidator()
        zip_path = temp_dir / "malicious_zipslip.zip"
        target_dir = temp_dir / "extracted"

        # Create zip with directory traversal
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../evil.py", "print('hacked')")

        with pytest.raises(ArchiveValidationError, match="ZipSlip"):
            validator.validate_and_extract(zip_path, target_dir)

    def test_blocked_extension_rejection(self, temp_dir):
        validator = ArchiveValidator()
        zip_path = temp_dir / "malicious_binary.zip"
        target_dir = temp_dir / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("malware.exe", "MZ...")

        with pytest.raises(ArchiveValidationError, match="Blocked file extension"):
            validator.validate_and_extract(zip_path, target_dir)

    def test_max_file_count_limit(self, temp_dir):
        validator = ArchiveValidator(max_files=5)
        zip_path = temp_dir / "too_many_files.zip"
        target_dir = temp_dir / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(10):
                zf.writestr(f"file_{i}.txt", f"data {i}")

        with pytest.raises(ArchiveValidationError, match="exceeds maximum"):
            validator.validate_and_extract(zip_path, target_dir)


class TestProjectInspector:
    """Tests for static AST analysis, framework detection, and security risk assessment."""

    def test_safe_agent_inspection(self, temp_dir):
        agent_file = temp_dir / "agent.py"
        agent_file.write_text(
            """
import json

def run(query: str):
    return {"response": f"Processed: {query}", "status": "SUCCESS"}
""",
            encoding="utf-8",
        )
        (temp_dir / "requirements.txt").write_text("pydantic>=2.0.0\n", encoding="utf-8")

        inspector = ProjectInspector()
        result = inspector.inspect_directory(temp_dir)

        assert result.overall_risk == SecurityRiskLevel.LOW
        assert result.is_safe_to_execute is True
        assert result.manifest.entry_point == "agent.py"
        assert result.manifest.entry_symbol == "run"
        assert "pydantic>=2.0.0" in result.manifest.dependencies

    def test_critical_eval_and_shell_detection(self, temp_dir):
        agent_file = temp_dir / "malicious_agent.py"
        agent_file.write_text(
            """
import os
import subprocess

def run(query: str):
    eval("print('danger')")
    os.system("rm -rf /")
    subprocess.Popen(["cat", "/etc/shadow"])
    return "done"
""",
            encoding="utf-8",
        )

        inspector = ProjectInspector()
        result = inspector.inspect_directory(temp_dir)

        assert result.overall_risk == SecurityRiskLevel.CRITICAL
        assert result.is_safe_to_execute is False
        assert result.rejection_reason is not None

        rule_ids = [a.rule_id for a in result.alerts]
        assert "DANGEROUS_CALL_EVAL" in rule_ids
        assert "DANGEROUS_API_os_system" in rule_ids
        assert "DANGEROUS_API_subprocess_Popen" in rule_ids

    def test_sensitive_credential_env_detection(self, temp_dir):
        agent_file = temp_dir / "agent.py"
        agent_file.write_text(
            """
import os

def run(query: str):
    key = os.environ.get("OPENAI_API_KEY")
    return "ready"
""",
            encoding="utf-8",
        )

        inspector = ProjectInspector()
        result = inspector.inspect_directory(temp_dir)

        assert "OPENAI_API_KEY" in result.manifest.required_env_vars
        assert any(a.rule_id == "ENV_KEY_REQUIRED" for a in result.alerts)


class TestSandboxIsolation:
    """Tests for environment sanitization and workspace lifecycle."""

    def test_host_secret_sanitization(self, temp_dir):
        # Set mock host secrets
        os.environ["HOST_OPENAI_API_KEY"] = "sk-super-secret-host-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret-host"
        os.environ["DATABASE_PASSWORD"] = "pass123"

        manager = SandboxEnvironmentManager(base_sandbox_dir=temp_dir)
        config = SandboxConfig(
            custom_env_vars={"ALLOWED_AGENT_KEY": "agent-token-xyz"},
        )

        workspace = temp_dir / "mock_workspace"
        workspace.mkdir()

        sanitized = manager.sanitize_environment(config, workspace)

        # Ensure host secrets were stripped
        assert "HOST_OPENAI_API_KEY" not in sanitized
        assert "AWS_SECRET_ACCESS_KEY" not in sanitized
        assert "DATABASE_PASSWORD" not in sanitized

        # Ensure custom allowed variable was injected
        assert sanitized.get("ALLOWED_AGENT_KEY") == "agent-token-xyz"
        assert sanitized.get("AGENT_EVAL_SANDBOX") == "1"

    def test_ephemeral_workspace_lifecycle(self, temp_dir):
        src_proj = temp_dir / "src_proj"
        src_proj.mkdir()
        (src_proj / "agent.py").write_text("def run(q): return 'ok'", encoding="utf-8")

        manager = SandboxEnvironmentManager(base_sandbox_dir=temp_dir / "sandboxes")
        ws = manager.create_ephemeral_workspace(src_proj)

        assert ws.is_dir()
        assert (ws / "agent.py").is_file()

        # Teardown
        cleaned = manager.cleanup_workspace(ws)
        assert cleaned is True
        assert not ws.exists()


class TestSubprocessSandboxDriverAndManager:
    """Tests for isolated process execution and safeguard limits."""

    def test_successful_sandboxed_execution(self, temp_dir):
        (temp_dir / "agent.py").write_text(
            """
def run(query: str):
    return {
        "response": f"Sandboxed echo: {query}",
        "steps": [
            {"step_type": "TOOL", "tool_name": "calculator", "content": "2+2", "output": "4", "status": "SUCCESS"}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
    }
""",
            encoding="utf-8",
        )

        manager = SandboxManager()
        result = manager.execute_task(
            project_dir=temp_dir,
            task_id="t1",
            query="Calculate budget",
            entry_point="agent.py",
            entry_symbol="run",
            config=SandboxConfig(timeout_seconds=5),
        )

        assert result.success is True
        assert result.status == "SUCCESS"
        assert "Sandboxed echo: Calculate budget" in result.response
        assert result.security_level == SecurityLevel.ISOLATED_PROCESS
        assert result.token_usage["total_tokens"] == 25
        assert len(result.trace_steps) == 1

    def test_execution_timeout_enforcement(self, temp_dir):
        (temp_dir / "slow_agent.py").write_text(
            """
import time

def run(query: str):
    time.sleep(5)
    return "done"
""",
            encoding="utf-8",
        )

        manager = SandboxManager()
        result = manager.execute_task(
            project_dir=temp_dir,
            task_id="timeout_test",
            query="hang",
            entry_point="slow_agent.py",
            entry_symbol="run",
            config=SandboxConfig(timeout_seconds=1),  # 1s timeout
        )

        assert result.success is False
        assert result.timed_out is True
        assert result.status == "TIMEOUT"
        assert "timed out" in result.response.lower()

    def test_output_truncation_enforcement(self, temp_dir):
        (temp_dir / "giant_agent.py").write_text(
            """
def run(query: str):
    return "A" * 10000
""",
            encoding="utf-8",
        )

        manager = SandboxManager()
        result = manager.execute_task(
            project_dir=temp_dir,
            task_id="trunc_test",
            query="generate",
            entry_point="giant_agent.py",
            entry_symbol="run",
            config=SandboxConfig(max_output_size_bytes=500),
        )

        assert len(result.response) <= 600
        assert "TRUNCATED" in result.response


class TestSandboxedAgentAdapter:
    """Tests for integration of SandboxedAgentAdapter with EvaluationEngine interfaces."""

    def test_adapter_trace_generation(self, temp_dir):
        (temp_dir / "agent.py").write_text(
            """
def run(query: str):
    return f"Response to {query}"
""",
            encoding="utf-8",
        )

        adapter = SandboxedAgentAdapter(
            agent_id="test_sandboxed_bot",
            project_dir=temp_dir,
            name="Test Bot",
            entry_point="agent.py",
            entry_symbol="run",
            sandbox_config=SandboxConfig(timeout_seconds=5),
        )

        res = adapter.run("Hello world", context={"task_id": "test_case_1"})

        assert res.success is True
        assert "Response to Hello world" in res.output
        assert res.trace.task_id == "test_case_1"
        assert len(res.trace.spans) >= 1
        assert res.metadata["security_level"] == SecurityLevel.ISOLATED_PROCESS.value

        # Test to_base_agent conversion
        base_agent = adapter.to_base_agent()
        assert base_agent.name == "Test Bot"
        test_case = TestCase(test_id="tc_99", user_input="Ping")
        trace = base_agent.run(test_case)
        assert trace.task_id == "tc_99"
        assert "Response to Ping" in trace.final_answer

    def test_registry_sandboxed_adapter_integration(self, temp_dir):
        from src.registry.registry import AgentRegistry
        from src.storage.db import init_db
        init_db()

        (temp_dir / "agent.py").write_text(
            """
def run(query: str):
    return f"Registry Sandboxed: {query}"
""",
            encoding="utf-8",
        )

        reg = AgentRegistry()
        unique_name = f"Sandboxed Agent {int(time.time() * 1000)}"
        agent = reg.register_agent(
            name=unique_name,
            description="Test sandboxed agent",
            integration_type="sandboxed",
            initial_version="v1.0",
            config={
                "type": "sandboxed",
                "project_dir": str(temp_dir),
                "entry_point": "agent.py",
                "entry_symbol": "run",
                "timeout_seconds": 5.0,
            },
        )

        adapter = reg.get_adapter(agent.agent_id)
        assert isinstance(adapter, SandboxedAgentAdapter)
        res = adapter.run("Evaluate me", context={"task_id": "reg_test_1"})
        assert res.success is True
        assert "Registry Sandboxed: Evaluate me" in res.output
