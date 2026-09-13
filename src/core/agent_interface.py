"""
Agent interface contract.

All agents evaluated by this framework implement BaseAgent, ensuring
complete decoupling between the agent runtime (LangGraph, CrewAI, AutoGen, REST API)
and downstream tracing, scoring, and analytics.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.core.entities import TestCase, Trace


class BaseAgent(ABC):
    """Abstract interface for any agent evaluated by the framework."""

    def __init__(
        self,
        name: str = "BaseAgent",
        version: str = "1.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.config = config or {}

    @abstractmethod
    def run(self, test_case: TestCase) -> Trace:
        """Execute a single TestCase and return an execution Trace."""
        pass

    def run_query(self, query: str, task_id: str = "adhoc") -> Trace:
        """Helper to run a raw query string directly without full TestCase wrapper."""
        test_case = TestCase(task_id=task_id, query=query)
        return self.run(test_case)
