"""
High-level evaluation runner for the AgentPulse SDK.

Allows external developers to evaluate any agent function or adapter
against golden datasets with a single command:
    report = agentpulse.evaluate(my_agent, dataset="golden_tasks")
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Optional, Union, List, Callable, Any, Dict

from src.core.entities import EvaluationDataset, TestCase, Trace, EvaluationReport
from src.core.agent_interface import BaseAgent
from src.core.evaluator_interface import BaseMetric
from src.core.engine import EvaluationEngine
from src.core.dataset import DatasetLoader
from src.registry.adapters import AgentAdapter, AgentExecutionResult, CallableAgentAdapter


def evaluate(
    agent: Union[Callable, BaseAgent, AgentAdapter],
    dataset: Union[str, EvaluationDataset, List[Dict[str, Any]]] = "golden_tasks",
    experiment_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_version: str = "v1.0",
    model: Optional[str] = None,
    metrics: Optional[List[BaseMetric]] = None,
    persist: bool = True,
    print_summary: bool = True,
) -> EvaluationReport:
    """Evaluate an agent against an evaluation dataset.

    Args:
        agent: Python callable, BaseAgent instance, or AgentAdapter.
        dataset: Dataset name (e.g. "golden_tasks"), filepath to JSON, or EvaluationDataset.
        experiment_name: Optional custom campaign name.
        agent_name: Optional name override.
        agent_version: Optional version tag.
        model: Optional model name.
        metrics: Optional custom list of evaluators (defaults to full standard suite).
        persist: Whether to persist runs, traces, and metrics into database.
        print_summary: Whether to print formatted pass/fail summary to terminal.

    Returns:
        EvaluationReport with overall scores, quality gate status, and metric breakdowns.
    """
    # 1. Resolve Agent
    if isinstance(agent, BaseAgent):
        base_agent = agent
    elif isinstance(agent, AgentAdapter):
        base_agent = agent.to_base_agent()
    elif callable(agent):
        adapter = CallableAgentAdapter(
            fn=agent,
            name=agent_name or getattr(agent, "__name__", "CustomAgent"),
            version=agent_version,
            model=model or "claude-3-5-haiku",
        )
        base_agent = adapter.to_base_agent()
    else:
        raise TypeError(f"Expected callable, BaseAgent, or AgentAdapter; got {type(agent)}")

    # 2. Resolve Dataset
    if isinstance(dataset, EvaluationDataset):
        eval_dataset = dataset
    elif isinstance(dataset, str):
        if dataset == "golden_tasks" or dataset.endswith(".json"):
            if dataset == "golden_tasks":
                path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dataset",
                    "golden_tasks.json",
                )
            else:
                path = dataset
            eval_dataset = DatasetLoader.from_json(path)
        else:
            # Look up dataset from DatasetManager
            from src.core.dataset_manager import DatasetManager
            mgr = DatasetManager()
            ds = mgr.get_dataset(dataset)
            if ds:
                eval_dataset = mgr.to_evaluation_dataset(dataset)
            else:
                raise ValueError(f"Dataset '{dataset}' not found as file or registered dataset.")
    elif isinstance(dataset, list):
        cases = [TestCase.from_dict(d) if isinstance(d, dict) else d for d in dataset]
        eval_dataset = EvaluationDataset(name="ad_hoc_dataset", test_cases=cases)
    else:
        raise TypeError(f"Invalid dataset type: {type(dataset)}")

    # 3. Configure and Execute Engine
    engine = EvaluationEngine(
        agent=base_agent,
        dataset=eval_dataset,
        metrics=metrics,
        experiment_name=experiment_name,
        persist=persist,
        model=model,
    )

    if print_summary:
        print(f"\n[AgentPulse SDK] Evaluating '{base_agent.name}' on '{eval_dataset.name}' ({len(eval_dataset.test_cases)} tests)...")

    def progress_callback(test_case: TestCase, trace: Trace, results: List[Any]):
        if print_summary:
            status = "✅ PASS" if all(r.passed for r in results) else "❌ FAIL"
            print(f"  [{test_case.task_id}] {status} | Latency: {trace.latency_ms:.0f}ms | Query: {test_case.query[:50]}")

    report = engine.run(progress_callback=progress_callback if print_summary else None)

    if print_summary:
        print("\n" + "=" * 60)
        print(f"Evaluation Complete: {report.experiment_name}")
        print(f"Overall Pass Rate : {report.overall_pass_rate:.1f}%")
        print(f"Quality Score     : {report.weighted_score:.1f} / 100.0")
        print(f"Passed Checks     : {report.passed_checks} / {report.total_checks}")
        print(f"Avg Latency       : {report.avg_latency_ms:.1f} ms")
        print(f"Estimated Cost    : ${report.estimated_cost_usd:.4f}")
        print("=" * 60 + "\n")

    return report
