"""
Configuration manager for the AgentPulse SDK.

Supports environment variables and programmatic overrides:
  - AGENTPULSE_BACKEND_URL (default: http://localhost:8000)
  - AGENTPULSE_API_KEY (default: "")
  - AGENTPULSE_TIMEOUT (default: 2.0s)
  - AGENTPULSE_DISABLED (default: false)
  - AGENTPULSE_AGENT_NAME (default: "DefaultAgent")
  - AGENTPULSE_AGENT_VERSION (default: "v1.0")
  - AGENTPULSE_ENV (default: "production")
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    backend_url: str = field(
        default_factory=lambda: os.getenv("AGENTPULSE_BACKEND_URL", "http://localhost:8000").rstrip("/")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("AGENTPULSE_API_KEY", "")
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("AGENTPULSE_TIMEOUT", "2.0"))
    )
    disabled: bool = field(
        default_factory=lambda: os.getenv("AGENTPULSE_DISABLED", "false").lower() in ("true", "1", "yes")
    )
    agent_name: str = field(
        default_factory=lambda: os.getenv("AGENTPULSE_AGENT_NAME", "DefaultAgent")
    )
    agent_version: str = field(
        default_factory=lambda: os.getenv("AGENTPULSE_AGENT_VERSION", "v1.0")
    )
    environment: str = field(
        default_factory=lambda: os.getenv("AGENTPULSE_ENV", "production")
    )


_global_config = Config()


def get_config() -> Config:
    """Return the active global configuration."""
    return _global_config


def configure(
    backend_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    disabled: Optional[bool] = None,
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
    environment: Optional[str] = None,
) -> Config:
    """Programmatically configure AgentPulse SDK settings."""
    global _global_config
    if backend_url is not None:
        _global_config.backend_url = backend_url.rstrip("/")
    if api_key is not None:
        _global_config.api_key = api_key
    if timeout is not None:
        _global_config.timeout = float(timeout)
    if disabled is not None:
        _global_config.disabled = bool(disabled)
    if agent_name is not None:
        _global_config.agent_name = str(agent_name)
    if agent_version is not None:
        _global_config.agent_version = str(agent_version)
    if environment is not None:
        _global_config.environment = str(environment)
    return _global_config


def reset_config() -> Config:
    """Reset configuration back to environment variable defaults."""
    global _global_config
    _global_config = Config()
    return _global_config
