"""
Public exports for the core Agent Evaluation & Observability Framework.
"""

from src.core.entities import (
    Agent,
    AgentVersion,
    EvaluationDataset,
    TestCase,
    Span,
    Trace,
    EvaluationResult,
    Experiment,
    EvaluationReport,
)
from src.core.agent_interface import BaseAgent
from src.core.metric_interface import BaseMetric
from src.core.dataset import DatasetLoader
from src.core.engine import EvaluationEngine
from src.core.test_case_manager import TestCaseManager

__all__ = [
    "Agent",
    "AgentVersion",
    "EvaluationDataset",
    "TestCase",
    "Span",
    "Trace",
    "EvaluationResult",
    "Experiment",
    "EvaluationReport",
    "BaseAgent",
    "BaseMetric",
    "DatasetLoader",
    "EvaluationEngine",
    "TestCaseManager",
]
