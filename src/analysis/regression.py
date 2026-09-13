"""
Version Comparison, Regression Detection, and Quality Gate Evaluation Engine.

Provides multi-dimensional version comparison across:
1. overall_score (Weighted Quality Composite)
2. task_success (Task Completion Rate)
3. tool_accuracy (Tool Selection & Execution Correctness)
4. groundedness (Context & Keyword Grounding)
5. answer_quality (Semantic & Exact Answer Quality)
6. latency (Execution Duration ms - Lower is Better)
7. token_usage (Total Tokens - Lower is Better)
8. cost (Estimated Cost USD - Lower is Better)
9. error_rate (Proportion of Failing Spans/Runs - Lower is Better)

Detects:
- Regressions (e.g. Task Success: 94% -> 86% = -8 pp, FAILED QUALITY GATE)
- Performance Improvements (e.g. Latency improved by 24%)
- Test Case Transitions (Newly Failing, Newly Passing, Persistent Failures)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import pandas as pd


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class TransitionType(str, Enum):
    NEWLY_FAILING = "NEWLY_FAILING"      # Regression: Passed in Baseline -> Failed in Target
    NEWLY_PASSING = "NEWLY_PASSING"      # Improvement: Failed in Baseline -> Passed in Target
    PERSISTENT_FAIL = "PERSISTENT_FAIL"  # Failed in both
    PERSISTENT_PASS = "PERSISTENT_PASS"  # Passed in both


@dataclass
class MetricComparison:
    """Side-by-side comparison for a single evaluation metric."""
    metric_name: str
    display_name: str
    baseline_value: float
    target_value: float
    delta: float                         # target - baseline
    pct_change: float                    # ((target - baseline) / baseline) * 100
    direction: Direction = Direction.HIGHER_IS_BETTER
    tolerance: float = 0.0               # Max acceptable regression threshold
    is_critical: bool = True             # If True, regression breaches Quality Gate
    unit: str = "%"                      # "%", "ms", "$", "tokens"
    is_regression: bool = False
    is_improvement: bool = False
    is_neutral: bool = True
    status_label: str = "Neutral"        # "Improved", "Regressed", "Neutral"
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "display_name": self.display_name,
            "baseline_value": round(self.baseline_value, 4),
            "target_value": round(self.target_value, 4),
            "delta": round(self.delta, 4),
            "pct_change": round(self.pct_change, 2),
            "direction": self.direction.value,
            "unit": self.unit,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
            "status_label": self.status_label,
            "explanation": self.explanation,
        }


@dataclass
class TestCaseDiff:
    """Diff transition analysis for an individual test case between versions."""
    task_id: str
    query: str
    baseline_passed: bool
    target_passed: bool
    baseline_score: float
    target_score: float
    transition_type: TransitionType
    baseline_error: Optional[str] = None
    target_error: Optional[str] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "baseline_passed": self.baseline_passed,
            "target_passed": self.target_passed,
            "baseline_score": round(self.baseline_score, 3),
            "target_score": round(self.target_score, 3),
            "transition_type": self.transition_type.value,
            "baseline_error": self.baseline_error,
            "target_error": self.target_error,
            "details": self.details,
        }


@dataclass
class QualityGatePolicy:
    """Configurable quality gate policy across metrics."""
    max_task_success_drop_pct: float = 0.0     # 0 tolerance for task success regression
    max_overall_score_drop_pct: float = 0.0    # 0 tolerance for overall score drop
    max_tool_accuracy_drop_pct: float = 0.0    # 0 tolerance for tool selection drop
    max_groundedness_drop_pct: float = 2.0     # max 2% groundedness tolerance
    max_latency_increase_pct: float = 15.0     # max 15% latency degradation before warning
    max_error_rate_increase_pct: float = 0.0   # 0 tolerance for new error spikes
    strict_critical_metrics: bool = True       # Enforce non-overall metric gates


@dataclass
class VersionComparisonReport:
    """Comprehensive version & experiment comparison report."""
    baseline_label: str
    target_label: str
    metrics: Dict[str, MetricComparison]
    quality_gate_passed: bool
    quality_gate_status: str                   # "PASSED QUALITY GATE" | "FAILED QUALITY GATE"
    quality_gate_reasons: List[str]
    improvement_highlights: List[str]
    regression_warnings: List[str]
    newly_failing_tests: List[TestCaseDiff]
    newly_passing_tests: List[TestCaseDiff]
    persistent_failing_tests: List[TestCaseDiff]
    persistent_passing_tests: List[TestCaseDiff]
    baseline_metadata: Dict[str, Any] = field(default_factory=dict)
    target_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "target_label": self.target_label,
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_status": self.quality_gate_status,
            "quality_gate_reasons": self.quality_gate_reasons,
            "improvement_highlights": self.improvement_highlights,
            "regression_warnings": self.regression_warnings,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "newly_failing_count": len(self.newly_failing_tests),
            "newly_passing_count": len(self.newly_passing_tests),
            "persistent_failing_count": len(self.persistent_failing_tests),
            "persistent_passing_count": len(self.persistent_passing_tests),
            "newly_failing_tests": [t.to_dict() for t in self.newly_failing_tests],
            "newly_passing_tests": [t.to_dict() for t in self.newly_passing_tests],
        }


class VersionComparator:
    """Engine for comparing two evaluation runs/experiments and detecting regressions."""

    @staticmethod
    def compare(
        baseline_runs: Union[pd.DataFrame, List[Dict[str, Any]]],
        target_runs: Union[pd.DataFrame, List[Dict[str, Any]]],
        baseline_evals: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        target_evals: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        baseline_label: str = "Baseline (v1)",
        target_label: str = "Target (v2)",
        policy: Optional[QualityGatePolicy] = None,
        baseline_meta: Optional[Dict[str, Any]] = None,
        target_meta: Optional[Dict[str, Any]] = None,
    ) -> VersionComparisonReport:
        """Perform full multi-metric comparison and quality gate verification."""
        pol = policy or QualityGatePolicy()

        # Convert to DataFrames if lists
        b_runs_df = pd.DataFrame(baseline_runs) if not isinstance(baseline_runs, pd.DataFrame) else baseline_runs.copy()
        t_runs_df = pd.DataFrame(target_runs) if not isinstance(target_runs, pd.DataFrame) else target_runs.copy()

        b_evals_df = (
            pd.DataFrame(baseline_evals) if (baseline_evals is not None and not isinstance(baseline_evals, pd.DataFrame))
            else (baseline_evals.copy() if isinstance(baseline_evals, pd.DataFrame) else pd.DataFrame())
        )
        t_evals_df = (
            pd.DataFrame(target_evals) if (target_evals is not None and not isinstance(target_evals, pd.DataFrame))
            else (target_evals.copy() if isinstance(target_evals, pd.DataFrame) else pd.DataFrame())
        )

        # -------------------------------------------------------------------
        # 1. Compute 9 Standard Comparison Metrics
        # -------------------------------------------------------------------
        def _compute_version_stats(runs_df: pd.DataFrame, evals_df: pd.DataFrame) -> Dict[str, float]:
            if runs_df.empty:
                return {
                    "overall_score": 0.0, "task_success": 0.0, "tool_accuracy": 0.0,
                    "groundedness": 0.0, "answer_quality": 0.0, "latency": 0.0,
                    "token_usage": 0.0, "cost": 0.0, "error_rate": 0.0,
                }

            # Helper for metric extraction from evals_df
            def _get_eval_metric_score(patterns: List[str], default_val: Optional[float] = None) -> float:
                if evals_df.empty or "metric_name" not in evals_df.columns:
                    return default_val if default_val is not None else 0.0
                mask = evals_df["metric_name"].apply(lambda m: any(p in str(m).lower() for p in patterns))
                subset = evals_df[mask]
                if subset.empty:
                    return default_val if default_val is not None else 0.0
                return float(subset["score"].mean() * 100.0)

            # Overall pass rate fallback
            overall_pass = float(evals_df["passed"].mean() * 100.0) if (not evals_df.empty and "passed" in evals_df.columns) else 0.0

            # 1. Overall Score
            overall_score = float(evals_df["score"].mean() * 100.0) if (not evals_df.empty and "score" in evals_df.columns) else overall_pass

            # 2. Task Success
            task_success = _get_eval_metric_score(["task_success", "success"], default_val=overall_pass)

            # 3. Tool Accuracy
            tool_accuracy = _get_eval_metric_score(["tool_selection", "tool_accuracy", "tool_argument"], default_val=100.0)

            # 4. Groundedness
            groundedness = _get_eval_metric_score(["groundedness", "context_groundedness", "keyword_groundedness", "faithfulness"], default_val=100.0)

            # 5. Answer Quality
            answer_quality = _get_eval_metric_score(["exact_answer", "semantic_answer", "similarity", "llm_judge", "quality"], default_val=overall_score)

            # 6. Latency (Average latency ms)
            avg_lat = float(runs_df["latency_ms"].mean()) if "latency_ms" in runs_df.columns else 0.0

            # 7. Token Usage (Average tokens per task)
            total_toks = 0.0
            if "input_tokens" in runs_df.columns and "output_tokens" in runs_df.columns:
                total_toks = float((runs_df["input_tokens"] + runs_df["output_tokens"]).mean())
            elif "total_tokens" in runs_df.columns:
                total_toks = float(runs_df["total_tokens"].mean())

            # 8. Cost (Average estimated cost USD per task)
            avg_cost = 0.0
            if "est_cost_usd" in runs_df.columns:
                avg_cost = float(runs_df["est_cost_usd"].mean())
            elif "cost_usd" in runs_df.columns:
                avg_cost = float(runs_df["cost_usd"].mean())
            elif total_toks > 0:
                from src.cost import global_cost_calculator
                model_name = runs_df["model"].iloc[0] if ("model" in runs_df.columns and not runs_df.empty) else "gpt-4o"
                pricing = global_cost_calculator.get_model_pricing(model_name)
                avg_cost = float(pricing.calculate_cost(int(total_toks * 0.6), int(total_toks * 0.4)))

            # 9. Error Rate (% of runs with failed status or errors)
            err_runs = 0
            if not evals_df.empty and "passed" in evals_df.columns:
                # Runs where any check failed or error occurred
                failed_runs = evals_df[evals_df["passed"] == False]["run_id"].nunique()
                err_runs = (failed_runs / len(runs_df) * 100.0) if len(runs_df) > 0 else 0.0

            return {
                "overall_score": overall_score,
                "task_success": task_success,
                "tool_accuracy": tool_accuracy,
                "groundedness": groundedness,
                "answer_quality": answer_quality,
                "latency": avg_lat,
                "token_usage": total_toks,
                "cost": avg_cost,
                "error_rate": err_runs,
            }

        b_stats = _compute_version_stats(b_runs_df, b_evals_df)
        t_stats = _compute_version_stats(t_runs_df, t_evals_df)

        # -------------------------------------------------------------------
        # 2. Build MetricComparisons & Detect Regressions/Improvements
        # -------------------------------------------------------------------
        metric_configs = [
            ("overall_score", "Overall Score", Direction.HIGHER_IS_BETTER, "%", pol.max_overall_score_drop_pct, True),
            ("task_success", "Task Success", Direction.HIGHER_IS_BETTER, "%", pol.max_task_success_drop_pct, True),
            ("tool_accuracy", "Tool Accuracy", Direction.HIGHER_IS_BETTER, "%", pol.max_tool_accuracy_drop_pct, True),
            ("groundedness", "Groundedness", Direction.HIGHER_IS_BETTER, "%", pol.max_groundedness_drop_pct, False),
            ("answer_quality", "Answer Quality", Direction.HIGHER_IS_BETTER, "%", 2.0, False),
            ("latency", "Latency", Direction.LOWER_IS_BETTER, "ms", pol.max_latency_increase_pct, False),
            ("token_usage", "Token Usage", Direction.LOWER_IS_BETTER, "tokens", 10.0, False),
            ("cost", "Cost per Run", Direction.LOWER_IS_BETTER, "$", 10.0, False),
            ("error_rate", "Error Rate", Direction.LOWER_IS_BETTER, "%", pol.max_error_rate_increase_pct, True),
        ]

        comparisons: Dict[str, MetricComparison] = {}
        quality_gate_reasons: List[str] = []
        improvements: List[str] = []
        regressions: List[str] = []

        for m_key, disp_name, direction, unit, tol, is_crit in metric_configs:
            b_val = b_stats.get(m_key, 0.0)
            t_val = t_stats.get(m_key, 0.0)
            delta = t_val - b_val
            pct_chg = ((t_val - b_val) / b_val * 100.0) if b_val != 0 else (0.0 if t_val == 0 else 100.0)

            is_regr = False
            is_impr = False
            is_neut = True
            expl = ""

            if direction == Direction.HIGHER_IS_BETTER:
                # Drop is bad
                if delta < -tol:
                    is_regr = True
                    is_neut = False
                    expl = f"{disp_name} regressed by {abs(delta):.1f} pp ({b_val:.1f}% ➔ {t_val:.1f}%)"
                    regressions.append(expl)
                    if is_crit and pol.strict_critical_metrics:
                        quality_gate_reasons.append(f"Critical regression in {disp_name}: dropped {abs(delta):.1f} percentage points.")
                elif delta > 0.5:
                    is_impr = True
                    is_neut = False
                    expl = f"{disp_name} improved by +{delta:.1f} pp ({b_val:.1f}% ➔ {t_val:.1f}%)"
                    improvements.append(expl)
                else:
                    expl = f"{disp_name} maintained at {t_val:.1f}%"
            else:
                # LOWER_IS_BETTER (Latency, Cost, Tokens, Error Rate)
                # Increase is bad
                if pct_chg > tol and (t_val > b_val + (0.0001 if unit == "$" else 1.0)):
                    is_regr = True
                    is_neut = False
                    expl = f"{disp_name} degraded by +{pct_chg:.1f}% ({b_val:.1f}{unit} ➔ {t_val:.1f}{unit})"
                    regressions.append(expl)
                    if is_crit and pol.strict_critical_metrics and m_key == "error_rate" and delta > 0:
                        quality_gate_reasons.append(f"Critical regression in {disp_name}: increased by +{delta:.1f} pp.")
                elif pct_chg < -1.0:
                    is_impr = True
                    is_neut = False
                    expl = f"{disp_name} improved by {abs(pct_chg):.1f}% ({b_val:.1f}{unit} ➔ {t_val:.1f}{unit})"
                    improvements.append(expl)
                else:
                    expl = f"{disp_name} stable ({t_val:.1f}{unit})"

            status_lbl = "Improved" if is_impr else ("Regressed" if is_regr else "Neutral")

            comparisons[m_key] = MetricComparison(
                metric_name=m_key,
                display_name=disp_name,
                baseline_value=b_val,
                target_value=t_val,
                delta=delta,
                pct_change=pct_chg,
                direction=direction,
                tolerance=tol,
                is_critical=is_crit,
                unit=unit,
                is_regression=is_regr,
                is_improvement=is_impr,
                is_neutral=is_neut,
                status_label=status_lbl,
                explanation=expl,
            )

        # -------------------------------------------------------------------
        # 3. Test Case Transitions (Newly Failing / Passing / Persistent)
        # -------------------------------------------------------------------
        newly_failing: List[TestCaseDiff] = []
        newly_passing: List[TestCaseDiff] = []
        persistent_failing: List[TestCaseDiff] = []
        persistent_passing: List[TestCaseDiff] = []

        # Group evals by task_id
        def _extract_task_info(grp: pd.DataFrame) -> Dict[str, Any]:
            passed = bool(grp["passed"].all()) if "passed" in grp.columns else True
            score = float(grp["score"].mean()) if "score" in grp.columns else 1.0
            err_col = "details" if "details" in grp.columns else ("explanation" if "explanation" in grp.columns else None)
            err_msg = ""
            if err_col and "passed" in grp.columns:
                failed_rows = grp[grp["passed"] == False][err_col].dropna().tolist()
                err_msg = "; ".join(str(e) for e in failed_rows[:1])
            return {"passed": passed, "score": score, "error": err_msg or None}

        b_tasks = {}
        if not b_evals_df.empty and "task_id" in b_evals_df.columns:
            for tid, grp in b_evals_df.groupby("task_id"):
                b_tasks[tid] = _extract_task_info(grp)

        t_tasks = {}
        if not t_evals_df.empty and "task_id" in t_evals_df.columns:
            for tid, grp in t_evals_df.groupby("task_id"):
                t_tasks[tid] = _extract_task_info(grp)

        all_task_ids = sorted(set(list(b_tasks.keys()) + list(t_tasks.keys())))

        # Look up queries from runs_df
        query_map = {}
        for df in [b_runs_df, t_runs_df]:
            if not df.empty and "task_id" in df.columns and "query" in df.columns:
                for _, row in df.iterrows():
                    query_map[row["task_id"]] = row["query"]

        for tid in all_task_ids:
            b_info = b_tasks.get(tid, {"passed": True, "score": 1.0, "error": None})
            t_info = t_tasks.get(tid, {"passed": True, "score": 1.0, "error": None})
            q_text = query_map.get(tid, f"Task {tid}")

            b_pass = b_info["passed"]
            t_pass = t_info["passed"]
            b_sc = b_info["score"]
            t_sc = t_info["score"]

            if b_pass and not t_pass:
                diff = TestCaseDiff(
                    task_id=tid, query=q_text,
                    baseline_passed=b_pass, target_passed=t_pass,
                    baseline_score=b_sc, target_score=t_sc,
                    transition_type=TransitionType.NEWLY_FAILING,
                    target_error=t_info["error"],
                    details=f"Regression: Passed in {baseline_label} (score {b_sc:.2f}) ➔ Failed in {target_label} (score {t_sc:.2f}).",
                )
                newly_failing.append(diff)
            elif not b_pass and t_pass:
                diff = TestCaseDiff(
                    task_id=tid, query=q_text,
                    baseline_passed=b_pass, target_passed=t_pass,
                    baseline_score=b_sc, target_score=t_sc,
                    transition_type=TransitionType.NEWLY_PASSING,
                    baseline_error=b_info["error"],
                    details=f"Improvement: Failed in {baseline_label} ➔ Now passing in {target_label} (score {t_sc:.2f}).",
                )
                newly_passing.append(diff)
            elif not b_pass and not t_pass:
                diff = TestCaseDiff(
                    task_id=tid, query=q_text,
                    baseline_passed=b_pass, target_passed=t_pass,
                    baseline_score=b_sc, target_score=t_sc,
                    transition_type=TransitionType.PERSISTENT_FAIL,
                    baseline_error=b_info["error"],
                    target_error=t_info["error"],
                    details=f"Persistent failure in both versions (v1: {b_sc:.2f}, v2: {t_sc:.2f}).",
                )
                persistent_failing.append(diff)
            else:
                diff = TestCaseDiff(
                    task_id=tid, query=q_text,
                    baseline_passed=b_pass, target_passed=t_pass,
                    baseline_score=b_sc, target_score=t_sc,
                    transition_type=TransitionType.PERSISTENT_PASS,
                    details=f"Consistently passed in both versions.",
                )
                persistent_passing.append(diff)

        # -------------------------------------------------------------------
        # 4. Final Quality Gate Verdict
        # -------------------------------------------------------------------
        # Fail gate if any critical regression exists or newly failing tests occurred
        if newly_failing:
            quality_gate_reasons.append(f"{len(newly_failing)} test case{'s' if len(newly_failing) != 1 else ''} newly failed in {target_label}.")

        gate_passed = (len(quality_gate_reasons) == 0)
        gate_status = "PASSED QUALITY GATE" if gate_passed else "FAILED QUALITY GATE"

        return VersionComparisonReport(
            baseline_label=baseline_label,
            target_label=target_label,
            metrics=comparisons,
            quality_gate_passed=gate_passed,
            quality_gate_status=gate_status,
            quality_gate_reasons=quality_gate_reasons,
            improvement_highlights=improvements,
            regression_warnings=regressions,
            newly_failing_tests=newly_failing,
            newly_passing_tests=newly_passing,
            persistent_failing_tests=persistent_failing,
            persistent_passing_tests=persistent_passing,
            baseline_metadata=baseline_meta or {},
            target_metadata=target_meta or {},
        )
