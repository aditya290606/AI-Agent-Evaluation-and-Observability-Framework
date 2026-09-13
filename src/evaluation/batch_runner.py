"""
Batch Evaluation Runner with Non-Blocking Asynchronous Execution.

Executes an Evaluation Dataset against an AI Agent in a background daemon thread,
tracking real-time progress, generating comprehensive Evaluation Run scorecards,
and allowing individual test/trace drill-downs without blocking the UI.
"""

import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd

from src.storage.db import get_session, init_db
from src.storage.models import Run, Step, EvalResult, Experiment as ExperimentRow
from src.core.entities import TestCase, Trace, EvaluationDataset, EvaluationResult
from src.core.agent_interface import BaseAgent
from src.core.evaluator_interface import BaseMetric
from src.evaluation.scoring import ScoringConfig, calculate_case_scores
from src.cost import global_cost_calculator
from src.security.redactor import SecretRedactor
from src.security.guardrails import ResourceGuardrails


@dataclass
class BatchJobState:
    """Live state of an ongoing or completed background batch evaluation job."""
    job_id: str
    experiment_id: str
    dataset_id: str
    dataset_name: str
    dataset_version: str
    agent_id: str
    agent_name: str
    agent_version: str
    model: str
    status: str = "pending"             # "pending" | "running" | "completed" | "failed" | "cancelled"
    total_tests: int = 0
    completed_tests: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    current_test_id: str = ""
    current_test_name: str = ""
    progress_pct: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False
    summary: Optional[Dict[str, Any]] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return max(0.0, end - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "status": self.status,
            "total_tests": self.total_tests,
            "completed_tests": self.completed_tests,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "current_test_id": self.current_test_id,
            "current_test_name": self.current_test_name,
            "progress_pct": self.progress_pct,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "error_message": self.error_message,
            "cancel_requested": self.cancel_requested,
            "has_summary": self.summary is not None,
        }


class BatchEvaluationRunner:
    """Orchestrates non-blocking background batch dataset evaluations."""

    _instance = None
    _lock = threading.Lock()
    _jobs: Dict[str, BatchJobState] = {}

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BatchEvaluationRunner, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        init_db()

    def start_batch_run(
        self,
        agent: BaseAgent,
        dataset: EvaluationDataset,
        scoring_config: Optional[ScoringConfig] = None,
        metrics: Optional[List[BaseMetric]] = None,
        model: Optional[str] = None,
        prompt_version: str = "v1.0",
        environment: str = "production",
    ) -> str:
        """
        Starts a dataset evaluation batch run asynchronously in a daemon thread.
        Returns the unique job_id immediately without blocking the caller.
        """
        job_id = f"batch_{uuid.uuid4().hex[:8]}"
        exp_id = f"eval_run_{uuid.uuid4().hex[:8]}"
        m_name = model or getattr(agent, "model_name", "claude-3-5-haiku")

        total = len(dataset.test_cases)
        job = BatchJobState(
            job_id=job_id,
            experiment_id=exp_id,
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            agent_id=getattr(agent, "agent_id", "") or getattr(agent, "id", "") or "demo",
            agent_name=agent.name,
            agent_version=getattr(agent, "version", "1.0"),
            model=m_name,
            status="pending",
            total_tests=total,
            completed_tests=0,
            passed_count=0,
            failed_count=0,
            skipped_count=0,
            progress_pct=0.0,
            start_time=time.time(),
        )

        with self._lock:
            self._jobs[job_id] = job

        # Spawn background daemon thread
        t = threading.Thread(
            target=self._execute_batch_thread,
            args=(job, agent, dataset, scoring_config, metrics, m_name, prompt_version, environment),
            daemon=True,
            name=f"BatchEvalThread-{job_id}"
        )
        t.start()
        return job_id

    def _execute_batch_thread(
        self,
        job: BatchJobState,
        agent: BaseAgent,
        dataset: EvaluationDataset,
        scoring_config: Optional[ScoringConfig],
        metrics: Optional[List[BaseMetric]],
        model: str,
        prompt_version: str,
        environment: str,
    ):
        """Worker thread executing tests in sequence, recording telemetry, and publishing progress."""
        cfg = scoring_config or ScoringConfig.balanced_preset()

        # Resolve metrics
        if metrics is None:
            from src.evaluation.evaluators import (
                TaskSuccessEvaluator,
                ToolSelectionAccuracyEvaluator,
                KeywordGroundednessEvaluator,
                LatencyBudgetEvaluator,
                LLMJudgeEvaluator,
            )
            eval_metrics = [
                TaskSuccessEvaluator(),
                ToolSelectionAccuracyEvaluator(),
                KeywordGroundednessEvaluator(),
                LatencyBudgetEvaluator(),
                LLMJudgeEvaluator(),
            ]
        else:
            eval_metrics = metrics

        job.status = "running"

        # Create parent Experiment record in SQLite
        session = get_session()
        try:
            exp_row = ExperimentRow(
                id=job.experiment_id,
                name=f"Evaluation Run: {dataset.name} ({dataset.version}) · {agent.name}",
                description=f"Batch evaluation of {len(dataset.test_cases)} test cases against {agent.name} ({job.agent_version})",
                agent_id=job.agent_id,
                agent_name=agent.name,
                agent_version=job.agent_version,
                model=model,
                prompt_version=prompt_version,
                dataset_name=dataset.name,
                dataset_version=dataset.version,
                environment=environment,
                status="running",
            )
            session.add(exp_row)
            session.commit()
        except Exception as e:
            job.status = "failed"
            job.error_message = f"Failed to initialize Evaluation Run record: {str(e)}"
            session.close()
            return

        total_tests = len(dataset.test_cases)
        all_case_summaries = []

        try:
            for idx, tc in enumerate(dataset.test_cases):
                if job.cancel_requested:
                    job.status = "cancelled"
                    break

                # Skip disabled tests
                if not getattr(tc, "enabled", True):
                    job.skipped_count += 1
                    job.completed_tests += 1
                    job.progress_pct = round((job.completed_tests / max(1, total_tests)) * 100, 1)
                    continue

                job.current_test_id = tc.test_id
                job.current_test_name = tc.name

                # 1. Execute agent
                try:
                    trace = agent.run(tc)
                except Exception as e:
                    trace = Trace(task_id=tc.task_id, query=tc.query, is_mock=False)
                    trace.finish(f"ERROR: {str(e)}")

                # 2. Run evaluators
                case_results: List[EvaluationResult] = []
                for metric in eval_metrics:
                    try:
                        res = metric(test_case=tc, execution_result=trace.final_answer, trace=trace)
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

                # Compute weighted scoring
                c_summary = calculate_case_scores(case_results, cfg)
                c_summary["test_id"] = tc.test_id
                c_summary["test_name"] = tc.name
                c_summary["latency_ms"] = trace.latency_ms
                all_case_summaries.append(c_summary)

                is_pass = c_summary.get("passed", False) and not c_summary.get("critical_failed", False)
                if is_pass:
                    job.passed_count += 1
                else:
                    job.failed_count += 1

                # 3. Persist to DB
                t_cost = global_cost_calculator.calculate_trace_cost(
                    trace, quality_score=c_summary.get("weighted_score"), default_model=model
                )
                run_row = Run(
                    experiment_id=job.experiment_id,
                    agent_id=job.agent_id,
                    agent_name=agent.name,
                    agent_version=job.agent_version,
                    model=model,
                    prompt_version=prompt_version,
                    dataset_version=dataset.version,
                    environment=environment,
                    task_id=tc.test_id,
                    query=SecretRedactor.redact_text(tc.user_input),
                    final_answer=ResourceGuardrails.truncate_output(SecretRedactor.redact_text(trace.final_answer)),
                    expected_tool=tc.expected_tool or "",
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

                # Persist Steps
                for step_idx, span in enumerate(trace.spans):
                    session.add(Step(
                        run_id=run_row.id,
                        step_index=step_idx,
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

                # Persist Eval Results
                for res in case_results:
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

                # Update progress
                job.completed_tests += 1
                job.progress_pct = round((job.completed_tests / max(1, total_tests)) * 100, 1)

            # Mark status
            if job.status != "cancelled":
                job.status = "completed"

            # Update Experiment record
            exp_rec = session.query(ExperimentRow).filter(ExperimentRow.id == job.experiment_id).first()
            if exp_rec:
                exp_rec.status = job.status
                session.commit()

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            session.rollback()
            try:
                exp_rec = session.query(ExperimentRow).filter(ExperimentRow.id == job.experiment_id).first()
                if exp_rec:
                    exp_rec.status = "failed"
                    session.commit()
            except Exception:
                pass
        finally:
            job.end_time = time.time()
            session.close()

            # Generate final summary scorecard
            try:
                job.summary = self.get_evaluation_run_summary(job.experiment_id)
            except Exception:
                job.summary = None

    def get_job(self, job_id: str) -> Optional[BatchJobState]:
        """Fetch job state by job_id."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job(self) -> Optional[BatchJobState]:
        """Fetch the most recently submitted batch evaluation job."""
        with self._lock:
            if not self._jobs:
                return None
            return sorted(self._jobs.values(), key=lambda j: j.start_time, reverse=True)[0]

    def cancel_job(self, job_id: str) -> bool:
        """Request immediate cancellation of an active batch job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ["pending", "running"]:
                job.cancel_requested = True
                return True
            return False

    def list_jobs(self) -> List[BatchJobState]:
        """List all tracked jobs, sorted newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.start_time, reverse=True)

    def get_evaluation_run_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive Evaluation Run summary scorecard and individual test results.

        Returns:
          - Total tests, Passed, Failed, Skipped
          - Overall score (weighted quality score)
          - Metric scores (pass rate and avg score per metric)
          - Average latency & P95 latency (ms)
          - Total cost ($)
          - Total tokens
          - test_runs: detailed list of each test execution with drill-down metadata
        """
        session = get_session()
        try:
            runs = session.query(Run).filter(Run.experiment_id == experiment_id).order_by(Run.id.asc()).all()
            if not runs:
                return {
                    "experiment_id": experiment_id,
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "overall_score": 0.0,
                    "metric_scores": {},
                    "average_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "total_tokens": 0,
                    "test_runs": [],
                }

            latencies = []
            tokens = []
            costs = []
            test_runs = []
            metric_accum: Dict[str, Dict[str, Any]] = {}
            weighted_scores = []
            passed_tests_count = 0
            failed_tests_count = 0

            for r in runs:
                lat = r.latency_ms or 0.0
                tok = (r.total_input_tokens or 0) + (r.total_output_tokens or 0)
                cost = r.est_cost_usd or 0.0

                latencies.append(lat)
                tokens.append(tok)
                costs.append(cost)

                # Process evaluations for this run
                evals_data = []
                all_passed = True
                scores = []
                for ev in r.eval_results:
                    evals_data.append({
                        "metric_name": ev.metric_name,
                        "score": ev.score,
                        "passed": ev.passed,
                        "details": ev.details,
                    })
                    if not ev.passed:
                        all_passed = False
                    scores.append(ev.score or 0.0)

                    # Accumulate metric statistics
                    if ev.metric_name not in metric_accum:
                        metric_accum[ev.metric_name] = {"passed": 0, "total": 0, "scores": []}
                    metric_accum[ev.metric_name]["total"] += 1
                    if ev.passed:
                        metric_accum[ev.metric_name]["passed"] += 1
                    metric_accum[ev.metric_name]["scores"].append(ev.score or 0.0)

                run_score = (sum(scores) / len(scores) * 100.0) if scores else 0.0
                weighted_scores.append(run_score)

                if all_passed and evals_data:
                    passed_tests_count += 1
                else:
                    failed_tests_count += 1

                test_runs.append({
                    "run_id": r.id,
                    "task_id": r.task_id,
                    "query": r.query,
                    "final_answer": r.final_answer,
                    "expected_tool": r.expected_tool,
                    "tools_called": r.tools_called,
                    "latency_ms": round(lat, 1),
                    "total_tokens": tok,
                    "cost_usd": round(cost, 5),
                    "score": round(run_score, 1),
                    "passed": all_passed,
                    "evaluations": evals_data,
                    "has_failures": not all_passed,
                })

            # P95 and Average latency
            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
            total_cost = sum(costs)
            total_tok = sum(tokens)
            overall_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0

            # Compute metric breakdown
            metric_scores: Dict[str, Dict[str, Any]] = {}
            for m_name, acc in metric_accum.items():
                m_pass_rate = (acc["passed"] / acc["total"] * 100.0) if acc["total"] else 0.0
                m_avg_score = (sum(acc["scores"]) / len(acc["scores"])) if acc["scores"] else 0.0
                metric_scores[m_name] = {
                    "passed": acc["passed"],
                    "total": acc["total"],
                    "pass_rate_pct": round(m_pass_rate, 1),
                    "avg_score": round(m_avg_score, 2),
                }

            # Check experiment row for dataset metadata
            exp_rec = session.query(ExperimentRow).filter(ExperimentRow.id == experiment_id).first()
            dataset_name = exp_rec.dataset_name if exp_rec else "Dataset"
            dataset_version = exp_rec.dataset_version if exp_rec else "1.0"
            agent_name = exp_rec.agent_name if exp_rec else "Agent"
            model_name = exp_rec.model if exp_rec else "claude-3-5-haiku"

            return {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "agent_name": agent_name,
                "model": model_name,
                "total_tests": len(runs),
                "passed": passed_tests_count,
                "failed": failed_tests_count,
                "skipped": 0,
                "overall_score": round(overall_score, 1),
                "metric_scores": metric_scores,
                "average_latency_ms": round(avg_lat, 1),
                "p95_latency_ms": round(p95_lat, 1),
                "total_cost_usd": round(total_cost, 5),
                "total_tokens": total_tok,
                "test_runs": test_runs,
            }
        finally:
            session.close()
