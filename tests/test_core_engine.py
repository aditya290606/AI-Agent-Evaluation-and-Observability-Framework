"""
Unit tests for the new core framework entities and evaluation engine.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.entities import (
    Agent,
    AgentVersion,
    TestCase,
    EvaluationDataset,
    Span,
    Trace,
    EvaluationResult,
    EvaluationReport,
)
from src.core.agent_interface import BaseAgent
from src.core.metric_interface import BaseMetric
from src.core.dataset import DatasetLoader
from src.core.engine import EvaluationEngine


class DummyEchoAgent(BaseAgent):
    """Simple test agent returning fixed outputs."""

    def run(self, test_case: TestCase) -> Trace:
        trace = Trace(task_id=test_case.task_id, query=test_case.query, is_mock=True)
        trace.log_span(Span(
            step_type="tool_call",
            tool_name="calculator",
            input_data="2 + 2",
            output_data="4",
            latency_ms=10.0,
        ))
        trace.finish("The calculated result is 4.")
        return trace


class AlwaysPassMetric(BaseMetric):
    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        return EvaluationResult(metric_name=self.name, score=1.0, passed=True, details="Always passes")


class ExactKeywordMetric(BaseMetric):
    def evaluate(self, trace: Trace, test_case: TestCase) -> EvaluationResult:
        has_kw = all(kw in trace.final_answer for kw in test_case.expected_keywords)
        return EvaluationResult(
            metric_name=self.name,
            score=1.0 if has_kw else 0.0,
            passed=has_kw,
            details=f"Keyword check: {has_kw}",
        )


def test_agent_and_version_creation():
    agent = Agent(name="TestAgent", description="Test agent")
    version = AgentVersion(version_tag="v1.0", model_name="claude-3-5-haiku")
    agent.add_version(version)

    assert agent.name == "TestAgent"
    assert len(agent.versions) == 1
    assert agent.versions[0].agent_id == agent.id


def test_test_case_and_dataset():
    case1 = TestCase(task_id="TC1", query="Query 1", expected_tool="calc", expected_keywords=["4"])
    case2 = TestCase(task_id="TC2", query="Query 2", tags=["math", "unit"])
    dataset = EvaluationDataset(name="math_tasks", test_cases=[case1, case2])

    assert len(dataset.test_cases) == 2
    assert len(dataset.filter_by_tags(["math"])) == 1


def test_dataset_loader_from_default():
    dataset = DatasetLoader.load_default_dataset()
    assert len(dataset.test_cases) == 15
    assert dataset.test_cases[0].task_id == "T001"


def test_trace_and_span_hierarchy():
    trace = Trace(task_id="T001", query="Hello")
    span1 = Span(step_type="llm_call", input_tokens=10, output_tokens=5)
    span2 = Span(step_type="tool_call", tool_name="search_knowledge_base", latency_ms=50.0)

    trace.log_span(span1)
    trace.log_span(span2)
    trace.finish("Final answer")

    assert trace.total_input_tokens == 10
    assert trace.total_output_tokens == 5
    assert trace.tools_called == ["search_knowledge_base"]
    assert len(trace.spans) == 2
    assert len(trace.steps) == 2  # backward compatibility alias


def test_evaluation_engine_execution():
    agent = DummyEchoAgent(name="EchoTestAgent")
    test_case = TestCase(task_id="TEST_1", query="Calculate 2+2", expected_keywords=["4"])
    dataset = EvaluationDataset(name="sample_dataset", test_cases=[test_case])
    metrics = [AlwaysPassMetric(name="always_pass"), ExactKeywordMetric(name="exact_keyword")]

    engine = EvaluationEngine(
        agent=agent,
        dataset=dataset,
        metrics=metrics,
        experiment_name="test_experiment",
        persist=False,  # in-memory test
    )

    report = engine.run()

    assert isinstance(report, EvaluationReport)
    assert report.total_test_cases == 1
    assert report.total_checks == 2
    assert report.passed_checks == 2
    assert report.overall_pass_rate == 100.0
    assert "always_pass" in report.metric_breakdown
    assert "exact_keyword" in report.metric_breakdown
