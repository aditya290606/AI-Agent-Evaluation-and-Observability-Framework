"""
Cost Analytics Engine for Agent Evaluation & Observability.

Computes aggregations, efficiency metrics, cost-vs-quality trade-offs,
and multi-dimensional breakdowns by agent, model, provider, and time.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


class CostAnalyticsEngine:
    """Computes advanced cost observability metrics across evaluation runs."""

    @staticmethod
    def compute_cost_summary(runs_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute top-level cost metrics from evaluation runs."""
        if runs_df.empty:
            return {
                "total_estimated_cost_usd": 0.0,
                "total_actual_cost_usd": 0.0,
                "has_actual_cost": False,
                "cost_per_run_avg": 0.0,
                "cost_per_successful_task": 0.0,
                "avg_cost_efficiency": 0.0,
                "total_tokens": 0,
                "total_runs": 0,
            }

        cost_col = "est_cost_usd" if "est_cost_usd" in runs_df.columns else "cost_usd"
        costs = runs_df[cost_col].fillna(0.0) if cost_col in runs_df.columns else pd.Series([0.0] * len(runs_df))
        total_est_cost = float(costs.sum())
        avg_cost_per_run = float(costs.mean())

        # Actual provider costs if present
        actual_cost_col = "actual_cost_usd" if "actual_cost_usd" in runs_df.columns else None
        total_actual_cost = float(runs_df[actual_cost_col].fillna(0.0).sum()) if actual_cost_col else 0.0
        has_actual_cost = bool(actual_cost_col and runs_df[actual_cost_col].notna().any() and total_actual_cost > 0)

        # Successful tasks count
        if "status" in runs_df.columns:
            successful_tasks = int((runs_df["status"] == "completed").sum())
        elif "passed" in runs_df.columns:
            successful_tasks = int((runs_df["passed"] == True).sum())
        else:
            successful_tasks = len(runs_df)

        cost_per_success = float(total_est_cost / max(1, successful_tasks))

        # Tokens
        total_toks = 0
        if "total_tokens" in runs_df.columns:
            total_toks = int(runs_df["total_tokens"].fillna(0).sum())
        elif "input_tokens" in runs_df.columns and "output_tokens" in runs_df.columns:
            total_toks = int((runs_df["input_tokens"].fillna(0) + runs_df["output_tokens"].fillna(0)).sum())

        # Cost efficiency: Score / Cost
        avg_efficiency = 0.0
        if "quality_score" in runs_df.columns and total_est_cost > 0:
            scores = runs_df["quality_score"].fillna(0.0)
            avg_score = float(scores.mean())
            avg_efficiency = float(avg_score / max(0.0001, avg_cost_per_run))
        elif "score" in runs_df.columns and total_est_cost > 0:
            avg_score = float(runs_df["score"].fillna(0.0).mean() * 100.0)
            avg_efficiency = float(avg_score / max(0.0001, avg_cost_per_run))

        return {
            "total_estimated_cost_usd": round(total_est_cost, 6),
            "total_actual_cost_usd": round(total_actual_cost, 6),
            "has_actual_cost": has_actual_cost,
            "cost_per_run_avg": round(avg_cost_per_run, 6),
            "cost_per_successful_task": round(cost_per_success, 6),
            "avg_cost_efficiency": round(avg_efficiency, 2),
            "total_tokens": total_toks,
            "total_runs": len(runs_df),
            "successful_tasks_count": successful_tasks,
        }

    @staticmethod
    def compute_cost_by_model(runs_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate total cost, token counts, and efficiency grouped by LLM model."""
        if runs_df.empty:
            return pd.DataFrame(columns=["model", "total_runs", "total_tokens", "total_cost_usd", "avg_cost_per_run", "cost_efficiency"])

        model_col = "model" if "model" in runs_df.columns else "model_name"
        if model_col not in runs_df.columns:
            runs_df = runs_df.copy()
            runs_df[model_col] = "gpt-4o"

        cost_col = "est_cost_usd" if "est_cost_usd" in runs_df.columns else "cost_usd"
        if cost_col not in runs_df.columns:
            runs_df = runs_df.copy()
            runs_df[cost_col] = 0.0

        records = []
        for model_name, group in runs_df.groupby(model_col):
            tot_cost = float(group[cost_col].fillna(0.0).sum())
            run_cnt = len(group)
            avg_cost = tot_cost / max(1, run_cnt)
            
            # Tokens
            if "total_tokens" in group.columns:
                toks = int(group["total_tokens"].fillna(0).sum())
            elif "input_tokens" in group.columns and "output_tokens" in group.columns:
                toks = int((group["input_tokens"].fillna(0) + group["output_tokens"].fillna(0)).sum())
            else:
                toks = 0

            # Quality Score
            score_col = "quality_score" if "quality_score" in group.columns else ("score" if "score" in group.columns else None)
            mean_score = float(group[score_col].mean() * (100.0 if group[score_col].max() <= 1.0 else 1.0)) if score_col else 85.0
            eff = round(mean_score / max(0.0001, avg_cost), 2) if avg_cost > 0 else 0.0

            records.append({
                "model": str(model_name or "unknown"),
                "total_runs": run_cnt,
                "total_tokens": toks,
                "total_cost_usd": round(tot_cost, 6),
                "avg_cost_per_run": round(avg_cost, 6),
                "mean_quality_score": round(mean_score, 2),
                "cost_efficiency": eff,
            })

        res_df = pd.DataFrame(records)
        return res_df.sort_values(by="total_cost_usd", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_cost_by_agent(runs_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate total cost, token counts, and efficiency grouped by Agent ID/Version."""
        if runs_df.empty:
            return pd.DataFrame(columns=["agent", "version", "total_runs", "total_cost_usd", "avg_cost_per_run", "cost_efficiency"])

        agent_col = "agent_name" if "agent_name" in runs_df.columns else ("agent_id" if "agent_id" in runs_df.columns else None)
        if not agent_col or agent_col not in runs_df.columns:
            runs_df = runs_df.copy()
            runs_df["agent_name"] = "Default Agent"
            agent_col = "agent_name"

        ver_col = "agent_version" if "agent_version" in runs_df.columns else None
        if not ver_col or ver_col not in runs_df.columns:
            runs_df = runs_df.copy()
            runs_df["agent_version"] = "v1.0"
            ver_col = "agent_version"

        cost_col = "est_cost_usd" if "est_cost_usd" in runs_df.columns else "cost_usd"
        if cost_col not in runs_df.columns:
            runs_df = runs_df.copy()
            runs_df[cost_col] = 0.0

        records = []
        for (agent_name, ver), group in runs_df.groupby([agent_col, ver_col]):
            tot_cost = float(group[cost_col].fillna(0.0).sum())
            run_cnt = len(group)
            avg_cost = tot_cost / max(1, run_cnt)

            score_col = "quality_score" if "quality_score" in group.columns else ("score" if "score" in group.columns else None)
            mean_score = float(group[score_col].mean() * (100.0 if group[score_col].max() <= 1.0 else 1.0)) if score_col else 85.0
            eff = round(mean_score / max(0.0001, avg_cost), 2) if avg_cost > 0 else 0.0

            records.append({
                "agent": str(agent_name),
                "version": str(ver),
                "total_runs": run_cnt,
                "total_cost_usd": round(tot_cost, 6),
                "avg_cost_per_run": round(avg_cost, 6),
                "mean_quality_score": round(mean_score, 2),
                "cost_efficiency": eff,
            })

        res_df = pd.DataFrame(records)
        return res_df.sort_values(by="total_cost_usd", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_cost_vs_quality(runs_df: pd.DataFrame, evals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract Cost vs Quality data points per task/run for Pareto efficiency analysis."""
        if runs_df.empty:
            return pd.DataFrame(columns=["task_id", "model", "cost_usd", "quality_score", "cost_efficiency", "latency_ms"])

        cost_col = "est_cost_usd" if "est_cost_usd" in runs_df.columns else "cost_usd"
        df = runs_df.copy()
        if cost_col not in df.columns:
            df[cost_col] = 0.0

        # Resolve run identifier column: run_id or id
        run_id_col = "run_id" if "run_id" in df.columns else ("id" if "id" in df.columns else None)

        # Merge or extract score
        if "quality_score" not in df.columns:
            if evals_df is not None and not evals_df.empty and "score" in evals_df.columns:
                eval_id_col = "run_id" if "run_id" in evals_df.columns else ("id" if "id" in evals_df.columns else None)
                if eval_id_col and run_id_col:
                    mean_scores = evals_df.groupby(eval_id_col)["score"].mean() * 100.0
                    df["quality_score"] = df[run_id_col].map(mean_scores).fillna(85.0)
                else:
                    df["quality_score"] = 85.0
            elif "score" in df.columns:
                df["quality_score"] = df["score"] * (100.0 if df["score"].max() <= 1.0 else 1.0)
            else:
                df["quality_score"] = 85.0

        df["cost_usd"] = df[cost_col].fillna(0.0)
        df["cost_efficiency"] = df.apply(
            lambda r: round(float(r["quality_score"] / max(0.00001, r["cost_usd"])), 2), axis=1
        )
        df["latency_ms"] = df["latency_ms"].fillna(0.0) if "latency_ms" in df.columns else 0.0
        df["model"] = df["model"].fillna("unknown") if "model" in df.columns else "unknown"
        
        if "task_id" not in df.columns:
            df["task_id"] = df[run_id_col] if run_id_col else [f"task_{i}" for i in range(len(df))]
        else:
            df["task_id"] = df["task_id"].fillna(df[run_id_col] if run_id_col else "task")

        cols = ["task_id", "model", "cost_usd", "quality_score", "cost_efficiency", "latency_ms"]
        return df[[c for c in cols if c in df.columns]].copy()
