"""
Core evaluation execution engine.

Orchestrates the formal lifecycle:
  Agent × TestCase → Trace (Spans) → Metrics → EvaluationResult → EvaluationReport
Persists execution runs and outputs comprehensive reports.
"""

import json
import time
from typing import List, Optional, Callable, Dict, Any
from src.core.entities import (
    TestCase,
    Trace,
    Span,
    EvaluationDataset,
    EvaluationResult,
    Experiment,
    EvaluationReport,
)
from src.core.agent_interface import BaseAgent
from src.core.evaluator_interface import BaseEvaluator, BaseMetric
from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, Experiment as ExperimentRow
from src.cost import global_cost_calculator, CostBreakdown


class EvaluationEngine:
    """Core evaluation orchestrator with modular evaluator architecture and weighted scoring."""

    def __init__(
        self,
        agent: BaseAgent,
        dataset: EvaluationDataset,
        metrics: Optional[List[BaseMetric]] = None,
        experiment_name: Optional[str] = None,
        scoring_config: Optional[ScoringConfig] = None,
        persist: bool = True,
        prompt_version: str = "v1.0",
        dataset_version: str = "v1.0",
        environment: str = "production",
        model: Optional[str] = None,
    ):
        self.agent = agent
        self.dataset = dataset
        self.scoring_config = scoring_config or ScoringConfig.balanced_preset()
        self.prompt_version = prompt_version
        self.dataset_version = dataset_version
        self.environment = environment
        self.model = model or getattr(agent, "model_name", "claude-3-5-haiku")

        if metrics is None:
            from src.evaluation.evaluators import (
                TaskSuccessEvaluator,
                ToolAccuracyEvaluator,
                AnswerCorrectnessEvaluator,
                GroundednessEvaluator,
                LatencyEvaluator,
            )
            self.metrics = [
                TaskSuccessEvaluator(),
                ToolAccuracyEvaluator(),
                AnswerCorrectnessEvaluator(),
                GroundednessEvaluator(),
                LatencyEvaluator(),
            ]
        else:
            self.metrics = metrics

        self.persist = persist
        self.experiment = Experiment(
            name=experiment_name or f"{agent.name}_{dataset.name}_{int(time.time())}",
            agent_id=getattr(agent, "agent_id", "") or getattr(agent, "id", ""),
            agent_name=agent.name,
            agent_version=agent.version,
            model=self.model,
            prompt_version=self.prompt_version,
            dataset_name=dataset.name,
            dataset_version=self.dataset_version,
            environment=self.environment,
        )

    def run(
        self,
        progress_callback: Optional[Callable[[TestCase, Trace, List[EvaluationResult]], None]] = None,
    ) -> EvaluationReport:
        """Execute the evaluation suite across all test cases."""
        if self.persist:
            init_db()
            session = get_session()
            exp_row = ExperimentRow(
                id=self.experiment.id,
                name=self.experiment.name,
                agent_id=self.experiment.agent_id,
                agent_name=self.experiment.agent_name,
                agent_version=self.experiment.agent_version,
                model=self.experiment.model,
                prompt_version=self.experiment.prompt_version,
                dataset_name=self.experiment.dataset_name,
                dataset_version=self.experiment.dataset_version,
                environment=self.experiment.environment,
                status="running",
            )
            session.add(exp_row)
            session.commit()
        else:
            session = None

        all_results: List[List[EvaluationResult]] = []
        all_traces: List[Trace] = []
        all_case_summaries: List[Dict[str, Any]] = []
        start_time = time.time()

        try:
            for test_case in self.dataset.test_cases:
                # 1. Execute agent with fault tolerance
                try:
                    trace = self.agent.run(test_case)
                except Exception as e:
                    trace = Trace(
                        task_id=test_case.task_id,
                        query=test_case.query,
                        is_mock=False,
                    )
                    trace.finish(f"ERROR: {str(e)}")

                all_traces.append(trace)

                # 2. Evaluate all configured evaluators / metrics
                case_results: List[EvaluationResult] = []
                for metric in self.metrics:
                    try:
                        res = metric(test_case=test_case, execution_result=trace.final_answer, trace=trace)
                    except Exception as e:
                        res = EvaluationResult(
                            metric_name=getattr(metric, "name", "metric"),
                            score=0.0,
                            passed=False,
                            threshold=getattr(metric, "threshold", 1.0),
                            explanation=f"Evaluator error: {str(e)}",
                            details=f"Evaluator error: {str(e)}",
                            evidence={"error": str(e)},
                            evaluator_type=getattr(metric, "evaluator_type", "deterministic"),
                        )
                    case_results.append(res)
                all_results.append(case_results)

                # Calculate individual scores, weighted composite score, and failure reasons
                case_summary = calculate_case_scores(case_results, self.scoring_config)
                case_summary["test_id"] = test_case.task_id

                # Failure Analysis for failed evaluation
                if not case_summary.get("passed", True) or any(not r.passed for r in case_results):
                    from src.analysis.failure_analysis import FailureAnalyzer
                    fa = FailureAnalyzer.analyze(test_case=test_case, trace=trace, eval_results=case_results)
                    case_summary["failure_analysis"] = fa.to_dict()
                else:
                    case_summary["failure_analysis"] = None

                all_case_summaries.append(case_summary)

                # 3. Persist run, steps/spans, and eval results
                if self.persist and session:
                    from src.security.redactor import SecretRedactor
                    from src.security.guardrails import ResourceGuardrails

                    t_cost = global_cost_calculator.calculate_trace_cost(
                        trace, quality_score=case_summary.get("weighted_score"), default_model=self.model
                    )
                    run_row = Run(
                        experiment_id=self.experiment.id,
                        agent_id=self.experiment.agent_id or getattr(self.agent, "agent_id", "") or "demo",
                        agent_name=self.experiment.agent_name,
                        agent_version=self.experiment.agent_version,
                        model=self.model,
                        prompt_version=self.prompt_version,
                        dataset_version=self.dataset_version,
                        environment=self.environment,
                        task_id=test_case.task_id,
                        query=SecretRedactor.redact_text(test_case.query),
                        final_answer=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(trace.final_answer)),
                        expected_tool=test_case.expected_tool or "",
                        tools_called=",".join(trace.tools_called),
                        total_input_tokens=trace.total_input_tokens,
                        total_output_tokens=trace.total_output_tokens,
                        latency_ms=trace.latency_ms,
                        est_cost_usd=t_cost.total_cost,
                        actual_cost_usd=t_cost.actual_cost,
                        is_mock=trace.is_mock,
                    )
                    session.add(run_row)
                    session.flush()

                    for idx, span in enumerate(trace.spans):
                        session.add(Step(
                            run_id=run_row.id,
                            step_index=idx,
                            step_type=span.step_type,
                            span_id=span.span_id,
                            parent_span_id=span.parent_span_id,
                            operation_name=SecretRedactor.redact_text(span.operation_name),
                            tool_name=span.tool_name,
                            input_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(span.input_data)),
                            output_data=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(span.output_data)),
                            latency_ms=span.latency_ms,
                            start_time=span.start_time,
                            end_time=span.end_time,
                            status=span.status or "success",
                            error=SecretRedactor.redact_text(span.error) if span.error else None,
                            model=span.model,
                            cost_usd=span.cost_usd,
                            metadata_json=json.dumps(SecretRedactor.redact_data(span.attributes or {})),
                            input_tokens=span.input_tokens,
                            output_tokens=span.output_tokens,
                        ))

                    for res in case_results:
                        res.run_id = run_row.id
                        session.add(EvalResult(
                            run_id=run_row.id,
                            metric_name=res.metric_name,
                            score=res.score,
                            passed=res.passed,
                            threshold=getattr(res, "threshold", 1.0),
                            evaluator_type=getattr(res, "evaluator_type", "deterministic"),
                            details=SecretRedactor.redact_text(res.explanation or res.details),
                            evidence_json=json.dumps(SecretRedactor.redact_data(getattr(res, "evidence", {}) or {})),
                        ))

                    session.commit()

                if progress_callback:
                    progress_callback(test_case, trace, case_results)

            if self.persist and session:
                # Mark experiment completed
                exp_row.status = "completed"
                session.commit()
        except Exception:
            if self.persist and session:
                session.rollback()
                try:
                    exp_row.status = "failed"
                    session.commit()
                except Exception:
                    pass
            raise
        finally:
            if self.persist and session:
                session.close()

        # 4. Synthesize EvaluationReport
        report = self._build_report(all_traces, all_results, all_case_summaries, time.time() - start_time)
        return report

    def _build_report(
        self,
        traces: List[Trace],
        results: List[List[EvaluationResult]],
        case_summaries: List[Dict[str, Any]],
        elapsed_seconds: float,
    ) -> EvaluationReport:
        total_checks = sum(len(r) for r in results)
        passed_checks = sum(sum(1 for res in r if res.passed) for r in results)
        overall_pass_rate = (passed_checks / total_checks * 100.0) if total_checks else 0.0

        # Per-metric breakdown
        metric_stats: Dict[str, Dict[str, Any]] = {}
        for r_list in results:
            for res in r_list:
                if res.metric_name not in metric_stats:
                    metric_stats[res.metric_name] = {"total": 0, "passed": 0, "scores": []}
                metric_stats[res.metric_name]["total"] += 1
                if res.passed:
                    metric_stats[res.metric_name]["passed"] += 1
                metric_stats[res.metric_name]["scores"].append(res.score)

        metric_breakdown: Dict[str, Dict[str, Any]] = {}
        for m_name, stats in metric_stats.items():
            pass_rate = (stats["passed"] / stats["total"] * 100.0) if stats["total"] else 0.0
            avg_score = (sum(stats["scores"]) / len(stats["scores"])) if stats["scores"] else 0.0
            metric_breakdown[m_name] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "pass_rate_pct": round(pass_rate, 2),
                "avg_score": round(avg_score, 3),
                "weight": self.scoring_config.weights.get(m_name, 1.0),
            }

        total_latency = sum(t.latency_ms for t in traces)
        avg_latency = (total_latency / len(traces)) if traces else 0.0
        total_in_tokens = sum(t.total_input_tokens for t in traces)
        total_out_tokens = sum(t.total_output_tokens for t in traces)
        total_tokens = total_in_tokens + total_out_tokens

        # Centralized Model/Provider-aware Cost Calculation
        trace_costs = [
            global_cost_calculator.calculate_trace_cost(t, default_model=self.model)
            for t in traces
        ]
        cost_usd = sum(tc.total_cost for tc in trace_costs) if trace_costs else 0.0

        # Aggregate weighted scores and failure reasons
        weighted_scores = [cs["weighted_score"] for cs in case_summaries]
        avg_weighted = (sum(weighted_scores) / len(weighted_scores)) if weighted_scores else 0.0

        overall_scores = [cs["overall_score"] for cs in case_summaries]
        avg_overall = (sum(overall_scores) / len(overall_scores)) if overall_scores else 0.0

        all_failures = []
        for cs in case_summaries:
            all_failures.extend(cs.get("failure_reasons", []))

        return EvaluationReport(
            experiment_id=self.experiment.id,
            experiment_name=self.experiment.name,
            agent_name=self.experiment.agent_name,
            dataset_name=self.experiment.dataset_name,
            total_test_cases=len(traces),
            total_checks=total_checks,
            passed_checks=passed_checks,
            overall_pass_rate=round(overall_pass_rate, 1),
            metric_breakdown=metric_breakdown,
            total_latency_ms=round(total_latency, 2),
            avg_latency_ms=round(avg_latency, 1),
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost_usd, 6),
            weighted_score=round(avg_weighted, 2),
            overall_score=round(avg_overall, 2),
            failure_reasons=all_failures,
            case_results=case_summaries,
            is_mock=any(t.is_mock for t in traces),
        )
