"""
Agent Registry service for registering, versioning, managing, and instantiating AI agents.
"""

import json
import uuid
from typing import List, Optional, Dict, Any

from src.storage.db import init_db, get_session
from src.storage.models import AgentRecord, AgentVersionRecord
from src.registry.adapters import (
    AgentAdapter,
    MockAgentAdapter,
    LocalPythonAdapter,
    HttpAgentAdapter,
)


def _gen_id(prefix: str = "agt") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class AgentRegistry:
    """Production Agent Registry managing persistent agent records, versions, and adapters."""

    def __init__(self):
        init_db()

    def register_agent(
        self,
        name: str,
        description: str = "",
        framework: str = "custom",
        provider_model: str = "claude-3-5-haiku",
        integration_type: str = "mock",
        initial_version: str = "v1.0",
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        status: str = "active",
    ) -> AgentRecord:
        """Register a new agent along with its initial version."""
        aid = agent_id or _gen_id("agt")
        cfg_str = json.dumps(config or {})

        session = get_session()
        try:
            # Check if name already exists
            existing = session.query(AgentRecord).filter(AgentRecord.name == name).first()
            if existing:
                raise ValueError(f"Agent with name '{name}' already exists (ID: {existing.agent_id})")

            agent = AgentRecord(
                agent_id=aid,
                name=name,
                description=description,
                framework=framework,
                provider_model=provider_model,
                integration_type=integration_type,
                status=status,
                active_version=initial_version,
            )
            version = AgentVersionRecord(
                version_id=_gen_id("ver"),
                agent_id=aid,
                version=initial_version,
                model=provider_model,
                configuration_metadata=cfg_str,
                status="active",
            )
            session.add(agent)
            session.add(version)
            session.commit()
            session.refresh(agent)
            return agent
        finally:
            session.close()

    def get_agent(self, agent_id_or_name: str) -> Optional[AgentRecord]:
        """Fetch an agent by its ID or unique name."""
        session = get_session()
        try:
            agent = session.query(AgentRecord).filter(
                (AgentRecord.agent_id == agent_id_or_name) | (AgentRecord.name == agent_id_or_name)
            ).first()
            if agent:
                _ = len(agent.versions)
            return agent
        finally:
            session.close()

    def get_agent_versions(self, agent_id: str) -> List[AgentVersionRecord]:
        """Fetch all version records for a given agent_id."""
        session = get_session()
        try:
            return session.query(AgentVersionRecord).filter(
                AgentVersionRecord.agent_id == agent_id
            ).order_by(AgentVersionRecord.created_at.desc()).all()
        finally:
            session.close()

    def list_agents(self, status: Optional[str] = None) -> List[AgentRecord]:
        """List all agents, optionally filtered by status ('active' or 'disabled')."""
        session = get_session()
        try:
            query = session.query(AgentRecord).order_by(AgentRecord.created_at.desc())
            if status:
                query = query.filter(AgentRecord.status == status)
            agents = query.all()
            # Eager load versions before session close
            for a in agents:
                _ = len(a.versions)
            return agents
        finally:
            session.close()

    def add_version(
        self,
        agent_id: str,
        version: str,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        set_active: bool = False,
    ) -> AgentVersionRecord:
        """Add a new version to an existing agent."""
        session = get_session()
        try:
            agent = session.query(AgentRecord).filter(AgentRecord.agent_id == agent_id).first()
            if not agent:
                raise ValueError(f"Agent with ID '{agent_id}' not found.")

            # Check if version already exists for this agent
            existing_ver = session.query(AgentVersionRecord).filter(
                AgentVersionRecord.agent_id == agent_id,
                AgentVersionRecord.version == version,
            ).first()
            if existing_ver:
                raise ValueError(f"Version '{version}' already exists for agent '{agent.name}'.")

            ver_record = AgentVersionRecord(
                version_id=_gen_id("ver"),
                agent_id=agent_id,
                version=version,
                model=model or agent.provider_model,
                configuration_metadata=json.dumps(config or {}),
                status="active",
            )
            session.add(ver_record)
            if set_active:
                agent.active_version = version
            session.commit()
            return ver_record
        finally:
            session.close()

    def set_active_version(self, agent_id: str, version: str) -> bool:
        """Update the default/active version for an agent."""
        session = get_session()
        try:
            agent = session.query(AgentRecord).filter(AgentRecord.agent_id == agent_id).first()
            if not agent:
                return False
            # Verify version exists
            ver = session.query(AgentVersionRecord).filter(
                AgentVersionRecord.agent_id == agent_id,
                AgentVersionRecord.version == version,
            ).first()
            if not ver:
                raise ValueError(f"Version '{version}' does not exist for agent '{agent.name}'.")

            agent.active_version = version
            session.commit()
            return True
        finally:
            session.close()

    def set_agent_status(self, agent_id: str, status: str) -> bool:
        """Enable ('active') or disable ('disabled') an agent."""
        if status not in ("active", "disabled"):
            raise ValueError(f"Status must be 'active' or 'disabled', got '{status}'")

        session = get_session()
        try:
            agent = session.query(AgentRecord).filter(AgentRecord.agent_id == agent_id).first()
            if not agent:
                return False
            agent.status = status
            session.commit()
            return True
        finally:
            session.close()

    def get_adapter(self, agent_id_or_name: str, version: Optional[str] = None) -> AgentAdapter:
        """Factory method: returns the instantiated AgentAdapter for an agent and version."""
        session = get_session()
        try:
            agent = session.query(AgentRecord).filter(
                (AgentRecord.agent_id == agent_id_or_name) | (AgentRecord.name == agent_id_or_name)
            ).first()
            if not agent:
                raise ValueError(f"Agent '{agent_id_or_name}' not found in registry.")

            if agent.status == "disabled":
                raise RuntimeError(f"Agent '{agent.name}' (ID: {agent.agent_id}) is currently disabled.")

            target_version = version or agent.active_version or "v1.0"
            ver_record = session.query(AgentVersionRecord).filter(
                AgentVersionRecord.agent_id == agent.agent_id,
                AgentVersionRecord.version == target_version,
            ).first()

            config: Dict[str, Any] = {}
            if ver_record and ver_record.configuration_metadata:
                try:
                    config = json.loads(ver_record.configuration_metadata)
                except Exception:
                    config = {}

            # Construct appropriate adapter
            itype = agent.integration_type.lower()
            if itype == "sandboxed" or config.get("type") == "sandboxed":
                from src.sandbox.adapter import SandboxedAgentAdapter
                from src.sandbox.models import SandboxConfig
                sb_cfg = SandboxConfig(
                    timeout_seconds=float(config.get("timeout_seconds", 30.0)),
                )
                return SandboxedAgentAdapter(
                    agent_id=agent.agent_id,
                    project_dir=config.get("project_dir", "."),
                    name=agent.name,
                    version=target_version,
                    entry_point=config.get("entry_point", "agent.py"),
                    entry_symbol=config.get("entry_symbol", "run"),
                    sandbox_config=sb_cfg,
                    config=config,
                )
            elif itype == "http_api":
                return HttpAgentAdapter(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    version=target_version,
                    config=config,
                )
            elif itype == "local_python":
                return LocalPythonAdapter(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    version=target_version,
                    config=config,
                )
            else:
                # Default / mock
                return MockAgentAdapter(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    version=target_version,
                    config=config,
                )
        finally:
            session.close()
