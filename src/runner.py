"""
Eval suite runner.

This is the script you run every time you change a prompt, swap a model,
or modify agent logic. It:

  1. Loads the golden task dataset
  2. Runs each task through the agent (mock or live)
  3. Computes all metrics against each run
  4. Persists everything to the database
  5. Prints a pass/fail summary to the terminal

Usage:
    python -m src.runner                  # mock mode, uses .env default
    python -m src.runner --live           # real Claude API calls
    python -m src.runner --inject-bug     # demo: deliberately break tool routing
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.security.env_validator import enforce_environment
enforce_environment()

# Ensure UTF-8 output on Windows so emoji characters (✅/❌) print correctly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.demo_agent import run_agent, DemoAgent
from src.evaluation import metrics, llm_judge
from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult
from src.core.entities import EvaluationDataset
from src.core.dataset import DatasetLoader
from src.core.engine import EvaluationEngine
from src.evaluation.metrics import ToolAccuracyMetric, KeywordGroundednessMetric, LatencyBudgetMetric
from src.evaluation.llm_judge import LLMJudgeGroundednessMetric

GOLDEN_TASKS_PATH = os.path.join(os.path.dirname(__file__), "dataset", "golden_tasks.json")


def load_golden_tasks():
    with open(GOLDEN_TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def persist_run(session, trace, expected_tool: str) -> Run:
    from src.security.redactor import SecretRedactor
    from src.security.guardrails import ResourceGuardrails

    run_row = Run(
        task_id=trace.task_id,
        query=SecretRedactor.redact_text(trace.query),
        final_answer=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(trace.final_answer)),
        expected_tool=expected_tool,
        tools_called=",".join(trace.tools_called),
        total_input_tokens=trace.total_input_tokens,
        total_output_tokens=trace.total_output_tokens,
        latency_ms=trace.latency_ms,
        is_mock=trace.is_mock,
    )
    session.add(run_row)
    session.flush()

    for idx, step in enumerate(trace.steps):
        session.add(Step(
            run_id=run_row.id,
            step_index=idx,
            step_type=step.step_type,
            tool_name=step.tool_name,
            input_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(step.input_data)),
            output_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(step.output_data)),
            latency_ms=step.latency_ms,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
        ))
    return run_row


def evaluate_and_persist(session, run_row, trace, task, use_mock):
    checks = []

    score, passed, details = metrics.tool_accuracy(trace, task.get("expected_tool", ""))
    checks.append(("tool_accuracy", score, passed, details, 1.0, "deterministic", {}))

    score, passed, details = metrics.keyword_groundedness(trace, task.get("expected_keywords", []))
    checks.append(("keyword_groundedness", score, passed, details, 1.0, "deterministic", {}))

    score, passed, details = metrics.latency_budget(trace)
    checks.append(("latency_budget", score, passed, details, 1.0, "budget", {}))

    score, passed, details = llm_judge.groundedness(trace, use_mock=use_mock)
    judge_label = "LLM Judge — MOCK" if use_mock else "LLM Judge — LIVE"
    checks.append(("llm_judge_groundedness", score, passed, details, 0.70, judge_label, {"mode": "mock" if use_mock else "live"}))

    for metric_name, score, passed, details, thresh, etype, evid in checks:
        session.add(EvalResult(
            run_id=run_row.id,
            metric_name=metric_name,
            score=score,
            passed=passed,
            details=details,
            threshold=thresh,
            evaluator_type=etype,
            evidence_json=json.dumps(evid),
        ))
    return [(m, s, p, d) for m, s, p, d, _, _, _ in checks]


def main():
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--live", action="store_true", help="Use real Claude API calls instead of mock mode.")
    parser.add_argument("--inject-bug", action="store_true",
                         help="Demo flag: deliberately misroute calculator queries (mock mode only).")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Path to custom evaluation dataset JSON file (defaults to golden_tasks.json).")
    parser.add_argument("--experiment", type=str, default=None,
                         help="Custom experiment name to tag this evaluation batch.")
    parser.add_argument("--min-pass-rate", type=float, default=None,
                         help="CI quality gate: fail with exit code 1 if pass rate drops below this percentage.")
    parser.add_argument("--agent", type=str, default=None,
                         help="Name or ID of registered agent to evaluate from the Agent Registry.")
    parser.add_argument("--agent-version", type=str, default=None,
                         help="Specific version tag to evaluate (defaults to active_version).")
    parser.add_argument("--tag", type=str, default=None,
                         help="Filter evaluation to test cases with this tag (e.g. 'math', 'rag', 'refund').")
    parser.add_argument("--test-id", type=str, default=None,
                         help="Run evaluation against a single test case ID (e.g. 'T001').")
    parser.add_argument("--callable", type=str, default=None,
                         help="External Python callable in format 'module:function' or 'module.function' to evaluate via SDK adapter.")
    args = parser.parse_args()

    use_mock = not args.live
    if not use_mock and args.inject_bug:
        print("--inject-bug only works in mock mode; ignoring.")

    # 1. Load dataset (from custom JSON, tag filter, single test ID, or default golden tasks)
    if args.dataset:
        dataset = DatasetLoader.from_json(args.dataset)
    elif args.tag or args.test_id:
        from src.core.test_case_manager import TestCaseManager
        mgr = TestCaseManager()
        if args.test_id:
            tc = mgr.get_test_case(args.test_id)
            if not tc:
                print(f"Error: Test case '{args.test_id}' not found in database.")
                sys.exit(1)
            dataset = EvaluationDataset(name=f"test_{args.test_id}", test_cases=[tc])
        else:
            dataset = mgr.to_evaluation_dataset(tag=args.tag, dataset_name=f"tag_{args.tag}")
    else:
        from src.core.test_case_manager import TestCaseManager
        mgr = TestCaseManager()
        enabled_cases = mgr.list_test_cases(enabled_only=True)
        if enabled_cases:
            dataset = mgr.to_evaluation_dataset(enabled_only=True, dataset_name="configured_suite")
        else:
            dataset = DatasetLoader.from_json(GOLDEN_TASKS_PATH)

    # 2. Configure Agent (from external callable, registry, or default DemoAgent)
    if args.callable:
        import importlib
        if ":" in args.callable:
            mod_name, fn_name = args.callable.split(":", 1)
        else:
            mod_name, fn_name = args.callable.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        target_fn = getattr(mod, fn_name)
        from src.sdk.evaluator import CallableAgentAdapter
        adapter = CallableAgentAdapter(
            fn=target_fn,
            name=args.agent or fn_name,
            version=args.agent_version or "v1.0",
        )
        agent = adapter.to_base_agent()
    elif args.agent:
        from src.registry.registry import AgentRegistry
        registry = AgentRegistry()
        adapter = registry.get_adapter(args.agent, version=args.agent_version)
        agent = adapter.to_base_agent()
    else:
        agent = DemoAgent(
            name="DemoReActAgent",
            use_mock=use_mock,
            inject_bug=(use_mock and args.inject_bug),
        )

    # 3. Configure Metrics
    eval_metrics = [
        ToolAccuracyMetric(),
        KeywordGroundednessMetric(),
        LatencyBudgetMetric(),
        LLMJudgeGroundednessMetric(use_mock=use_mock),
    ]

    # 4. Initialize EvaluationEngine
    engine = EvaluationEngine(
        agent=agent,
        dataset=dataset,
        metrics=eval_metrics,
        experiment_name=args.experiment,
        persist=True,
    )

    print(f"\nRunning eval suite: {len(dataset.test_cases)} tasks | dataset={dataset.name} | mode={'MOCK' if use_mock else 'LIVE'}"
          f"{' | BUG INJECTED' if (use_mock and args.inject_bug) else ''}\n")
    print(f"{'TASK':<7}{'TOOL CHECK':<12}{'GROUNDED':<11}{'LATENCY':<10}{'JUDGE':<8}QUERY")
    print("-" * 100)

    failed_eval_analyses = []

    def print_progress(test_case, trace, results):
        flags = ["✅" if res.passed else "❌" for res in results]
        print(f"{test_case.task_id:<7}{flags[0]:<12}{flags[1]:<11}{flags[2]:<10}{flags[3]:<8}{test_case.query[:50]}")
        if any(not res.passed for res in results):
            from src.analysis.failure_analysis import FailureAnalyzer
            fa = FailureAnalyzer.analyze(test_case, trace, results)
            failed_eval_analyses.append((test_case.task_id, fa))

    report = engine.run(progress_callback=print_progress)

    print("-" * 100)
    print(f"\nExperiment: {report.experiment_name} (ID: {report.experiment_id})")
    print(f"Overall: {report.passed_checks}/{report.total_checks} checks passed ({report.overall_pass_rate:.1f}%) | Weighted Quality Score: {report.weighted_score:.1f}%")

    if failed_eval_analyses:
        print("\n" + "=" * 80)
        print(f"🔬 FAILURE ANALYSIS ({len(failed_eval_analyses)} Failed Evaluations)")
        print("=" * 80)
        for tid, fa in failed_eval_analyses:
            print(f"\n--- Task ID: {tid} ---")
            print(fa.format_text(include_disclaimer=True))
        print("\n" + "=" * 80)

    if report.failure_reasons:
        print(f"\nIdentified Failure Metric Triggers ({len(report.failure_reasons)}):")
        for r in report.failure_reasons[:5]:
            print(f"  - {r}")
    print(f"Total Latency: {report.total_latency_ms:.0f}ms | Avg: {report.avg_latency_ms:.1f}ms | Est Cost: ${report.estimated_cost_usd:.4f}")
    print("Run `streamlit run dashboard/app.py` to explore results visually.\n")

    if args.min_pass_rate is not None:
        if report.overall_pass_rate < args.min_pass_rate:
            print(f"❌ CI Gate FAILED: Pass rate {report.overall_pass_rate:.1f}% is below required {args.min_pass_rate:.1f}%")
            sys.exit(1)
        else:
            print(f"✅ CI Gate PASSED: Pass rate {report.overall_pass_rate:.1f}% meets threshold {args.min_pass_rate:.1f}%")


if __name__ == "__main__":
    main()

