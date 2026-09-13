"""
Tests for Evaluation Report Generator and 17-Section Audit Engine.
"""

import json
import os
import sys
import pytest
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, Experiment as ExperimentRow
from src.evaluation.scoring import ScoringConfig
from src.analysis.report_generator import (
    EvaluationReportGenerator,
    EvaluationReportData,
    ExecutiveSummary,
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Seed test experiment and runs for report generation tests."""
    init_db()
    session = get_session()
    try:
        # Clean up existing test records if any
        session.query(ExperimentRow).filter(ExperimentRow.id.in_(["exp_rep_base_001", "exp_rep_curr_002"])).delete(synchronize_session=False)
        session.query(Run).filter(Run.experiment_id.in_(["exp_rep_base_001", "exp_rep_curr_002"])).delete(synchronize_session=False)
        session.commit()

        # Create baseline experiment
        exp_base = ExperimentRow(
            id="exp_rep_base_001",
            name="Baseline Benchmark v1.0",
            agent_id="agent_demo",
            dataset_name="Golden Benchmark",
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        session.add(exp_base)

        # Baseline run
        run_b1 = Run(
            task_id="TASK-001",
            agent_name="DemoAgent",
            agent_version="v1.0",
            model="claude-3-5-haiku-20241022",
            experiment_id="exp_rep_base_001",
            dataset_version="1.0",
            latency_ms=1200.0,
            est_cost_usd=0.0015,
            total_input_tokens=150,
            total_output_tokens=50,
            query="Analyze AAPL Q3 revenue growth",
            created_at=datetime.now(timezone.utc),
        )
        session.add(run_b1)

        # Create target experiment
        exp_curr = ExperimentRow(
            id="exp_rep_curr_002",
            name="Target Evaluation Suite v1.4",
            agent_id="agent_demo",
            dataset_name="Customer Support Suite",
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        session.add(exp_curr)

        # Target run 1 (Passed)
        run_c1 = Run(
            task_id="TASK-001",
            agent_name="Customer Support Agent",
            agent_version="v1.4",
            model="claude-3-5-haiku-20241022",
            experiment_id="exp_rep_curr_002",
            dataset_version="1.4",
            latency_ms=1100.0,
            est_cost_usd=0.0018,
            total_input_tokens=200,
            total_output_tokens=60,
            query="Process customer refund request for invoice #1042",
            created_at=datetime.now(timezone.utc),
        )
        session.add(run_c1)

        # Target run 2 (Failed tool argument)
        run_c2 = Run(
            task_id="TASK-002",
            agent_name="Customer Support Agent",
            agent_version="v1.4",
            model="claude-3-5-haiku-20241022",
            experiment_id="exp_rep_curr_002",
            dataset_version="1.4",
            latency_ms=4800.0,
            est_cost_usd=0.0042,
            total_input_tokens=400,
            total_output_tokens=120,
            query="Update billing address with invalid zip code",
            created_at=datetime.now(timezone.utc),
        )
        session.add(run_c2)

        session.commit()

        # Add steps / spans
        step_1 = Step(
            run_id=run_c1.id,
            step_index=1,
            step_type="tool_call",
            tool_name="refund_lookup",
            latency_ms=300.0,
            status="success",
            cost_usd=0.0002,
        )
        step_2 = Step(
            run_id=run_c1.id,
            step_index=2,
            step_type="retriever",
            operation_name="rag_kb_search",
            latency_ms=250.0,
            status="success",
            cost_usd=0.0001,
        )
        step_3 = Step(
            run_id=run_c2.id,
            step_index=1,
            step_type="tool_call",
            tool_name="update_address",
            latency_ms=450.0,
            status="error",
            error="Validation failed: missing country_code",
            cost_usd=0.0003,
        )
        session.add_all([step_1, step_2, step_3])

        # Add eval results
        eval_1 = EvalResult(
            run_id=run_c1.id,
            metric_name="task_success",
            score=1.0,
            passed=True,
            threshold=1.0,
            evaluator_type="deterministic",
            details="Refund processed successfully",
        )
        eval_2 = EvalResult(
            run_id=run_c1.id,
            metric_name="tool_selection_accuracy",
            score=1.0,
            passed=True,
            threshold=1.0,
            evaluator_type="deterministic",
            details="Correct tool selected",
        )
        eval_3 = EvalResult(
            run_id=run_c1.id,
            metric_name="keyword_groundedness",
            score=0.92,
            passed=True,
            threshold=0.8,
            evaluator_type="deterministic",
            details="Grounded in context",
        )
        eval_4 = EvalResult(
            run_id=run_c2.id,
            metric_name="task_success",
            score=0.0,
            passed=False,
            threshold=1.0,
            evaluator_type="deterministic",
            details="Address update failed",
        )
        eval_5 = EvalResult(
            run_id=run_c2.id,
            metric_name="tool_argument_correctness",
            score=0.0,
            passed=False,
            threshold=1.0,
            evaluator_type="deterministic",
            details="Missing country_code in arguments",
        )
        session.add_all([eval_1, eval_2, eval_3, eval_4, eval_5])
        session.commit()

    finally:
        session.close()


def test_generate_report_data_all_17_sections():
    """Verify that generate_report_data accurately populates all 17 standard sections."""
    report = EvaluationReportGenerator.generate_report_data(
        experiment_id="exp_rep_curr_002",
        baseline_experiment_id="exp_rep_base_001",
    )

    assert isinstance(report, EvaluationReportData)
    assert report.report_id.startswith("REP-")
    assert report.experiment_name == "Target Evaluation Suite v1.4"

    # 1. Agent information
    assert report.sec1_agent_info.name == "Customer Support Agent"
    assert report.sec1_agent_info.status == "active"

    # 2. Agent version
    assert report.sec2_agent_version.version == "v1.4"

    # 3. Model
    assert "claude" in report.sec3_model_info.primary_model.lower()
    assert report.sec3_model_info.provider == "Anthropic"

    # 4. Dataset
    assert report.sec4_dataset_info.name == "Customer Support Suite"
    assert report.sec4_dataset_info.total_test_cases == 2

    # 5. Evaluation timestamp
    assert report.sec5_timestamp_info.duration_seconds > 0

    # 6. Overall score
    assert report.sec6_overall_score.score > 0
    assert "/ 100" in report.sec6_overall_score.score_display

    # 7. Quality gate status
    assert report.sec7_quality_gate.status in ["PASSED", "FAILED"]
    assert len(report.sec7_quality_gate.rules_evaluated) >= 3

    # 8. Metric breakdown
    assert len(report.sec8_metric_breakdown) >= 3
    metric_names = [m.metric_name for m in report.sec8_metric_breakdown]
    assert "task_success" in metric_names
    assert "tool_selection_accuracy" in metric_names

    # 9. Passed tests
    assert len(report.sec9_passed_tests) == 1
    assert report.sec9_passed_tests[0].task_id == "TASK-001"
    assert report.sec9_passed_tests[0].status == "passed"

    # 10. Failed tests
    assert len(report.sec10_failed_tests) == 1
    assert report.sec10_failed_tests[0].task_id == "TASK-002"
    assert report.sec10_failed_tests[0].status == "failed"

    # 11. Regression analysis
    assert report.sec11_regression_analysis.has_baseline is True
    assert "Baseline" in report.sec11_regression_analysis.baseline_name

    # 12. Latency analysis
    assert report.sec12_latency_analysis.mean_latency_ms > 0
    assert report.sec12_latency_analysis.p95_latency_ms > 0

    # 13. Cost analysis
    assert report.sec13_cost_analysis.total_cost_usd > 0
    assert report.sec13_cost_analysis.total_tokens > 0

    # 14. Tool analysis
    assert report.sec14_tool_analysis.total_tool_calls >= 2
    assert "refund_lookup" in report.sec14_tool_analysis.unique_tools_used or "update_address" in report.sec14_tool_analysis.unique_tools_used

    # 15. RAG analysis
    assert report.sec15_rag_analysis.total_retrievals >= 1
    assert report.sec15_rag_analysis.answer_faithfulness_pct > 0

    # 16. Root-cause analysis
    assert report.sec16_root_cause_analysis.critical_failures_count == 1
    assert len(report.sec16_root_cause_analysis.observed_facts_summary) >= 3

    # 17. Recommendations
    assert len(report.sec17_recommendations) >= 1
    rec_priorities = [r.priority for r in report.sec17_recommendations]
    assert any("P0" in p or "P1" in p or "P2" in p for p in rec_priorities)


def test_executive_summary_structure():
    """Verify executive summary contains all requested top-level fields."""
    report = EvaluationReportGenerator.generate_report_data(
        experiment_id="exp_rep_curr_002",
    )
    s = report.summary
    assert s.agent_name == "Customer Support Agent"
    assert s.version == "v1.4"
    assert s.overall_score > 0
    assert s.quality_gate_status in ["PASSED", "FAILED"]
    assert 0 <= s.task_success_pct <= 100
    assert 0 <= s.tool_accuracy_pct <= 100
    assert 0 <= s.groundedness_pct <= 100
    assert s.p95_latency_sec > 0
    assert s.estimated_cost_per_run > 0
    assert s.critical_failures_count == 1
    assert s.total_tests == 2


def test_export_json_format():
    """Verify JSON export contains valid nested payload."""
    report = EvaluationReportGenerator.generate_report_data(
        experiment_id="exp_rep_curr_002",
    )
    json_str = EvaluationReportGenerator.export_json(report)
    parsed = json.loads(json_str)

    assert "report_id" in parsed
    assert "summary" in parsed
    assert "sec1_agent_info" in parsed
    assert "sec17_recommendations" in parsed
    assert parsed["summary"]["agent_name"] == "Customer Support Agent"


def test_export_csv_format():
    """Verify CSV export produces valid CSV datasets."""
    report = EvaluationReportGenerator.generate_report_data(
        experiment_id="exp_rep_curr_002",
    )
    csv_dict = EvaluationReportGenerator.export_csv(report)

    assert "summary.csv" in csv_dict
    assert "test_runs.csv" in csv_dict
    assert "metrics.csv" in csv_dict

    # Verify parsing back
    import io
    df_runs = pd.read_csv(io.StringIO(csv_dict["test_runs.csv"]))
    assert len(df_runs) == 2
    assert "TASK-001" in df_runs["task_id"].values
    assert "TASK-002" in df_runs["task_id"].values


def test_export_html_standalone_and_print_ready():
    """Verify HTML export produces standalone document with print-ready rules and all sections."""
    report = EvaluationReportGenerator.generate_report_data(
        experiment_id="exp_rep_curr_002",
    )
    html_content = EvaluationReportGenerator.export_html(report)

    assert "<!DOCTYPE html>" in html_content
    assert "@media print" in html_content
    assert "Customer Support Agent" in html_content
    assert "v1.4" in html_content
    assert "TASK-001" in html_content
    assert "TASK-002" in html_content
    assert "EXECUTIVE SUMMARY" in html_content
    assert "Actionable Engineering Recommendations" in html_content
