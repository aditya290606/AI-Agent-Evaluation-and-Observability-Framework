"""
Unit tests for environment security validation gate.
"""

import tempfile
from pathlib import Path
import pytest

from src.security.env_validator import check_environment, REQUIRED_KEYS


def test_missing_env_file_fails():
    is_valid, msg, details = check_environment(Path("does_not_exist.env"))
    assert not is_valid
    assert "was not found" in msg
    assert details["file_exists"] is False


def test_empty_env_file_fails():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write("# empty file\n")
        tmp_path = Path(f.name)

    try:
        is_valid, msg, details = check_environment(tmp_path)
        assert not is_valid
        assert "missing required security keys" in msg
        for k in REQUIRED_KEYS:
            assert k in details["missing_keys"]
    finally:
        tmp_path.unlink()


def test_placeholder_env_file_fails():
    content = (
        "AGENTPULSE_AUTH_KEY=your_key_here\n"
        "AGENTPULSE_API_KEY=your_api_key\n"
        "ANTHROPIC_API_KEY=your-anthropic-key\n"
        "DATABASE_URL=sqlite:///./test.db\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        is_valid, msg, details = check_environment(tmp_path)
        assert not is_valid
        assert "placeholders detected" in msg
    finally:
        tmp_path.unlink()


def test_authorized_env_file_passes():
    content = (
        "AGENTPULSE_AUTH_KEY=ap_sec_9f83a4c172e904b51829e847d3c01fa629471e892c50ab13e7d6c5b4a3928170\n"
        "AGENTPULSE_API_KEY=ap_live_7c8d9e2f1a0b3c4d5e6f7a8b9c0d1e2f\n"
        "ANTHROPIC_API_KEY=sk-ant-api03-testkey-1234567890\n"
        "DATABASE_URL=sqlite:///./eval_framework.db\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        is_valid, msg, details = check_environment(tmp_path)
        assert is_valid
        assert "verified" in msg
        assert len(details["missing_keys"]) == 0
    finally:
        tmp_path.unlink()
