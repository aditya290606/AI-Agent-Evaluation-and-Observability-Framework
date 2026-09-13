"""
Automated unit and integration tests for AI Evaluation Copilot.
"""

import os
import sys
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, Experiment as ExperimentRow
from src.analysis.copilot import EvaluationCopilot, CopilotResponse, CopilotFinding


@pytest.fixture(scope="module", autouse=True)
def setup_copilot_test_data():
    """Seed comprehensive evaluation data for testing Copilot analytics."""
    init_db()
    session = get_session()
    try:
        # Clean up prior test records if any
        session.query(ExperimentRow).filter(ExperimentRow.id.in_(["exp_copilot_v1", "exp_copilot_v2"])).delete(synchronize_session=False)
        session.query(Run).filter(Run.experiment_id.in_(["exp_copilot_v1", "exp_copilot_v2"])).delete(synchronize_session=False)
        session.commit()

        # 1. Experiment v1
        exp_v1 = ExperimentRow(
            id="exp_copilot_v1",
            name="Copilot Suite v1.0",
            agent_id="agent_copilot",
            dataset_name="Golden Copilot Tasks",
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        # 2. Experiment v2
        exp_v2 = ExperimentRow(
            id="exp_copilot_v2",
            name="Copilot Suite v2.0",
            agent_id="agent_copilot",
            dataset_name="Golden Copilot Tasks",
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([exp_v1, exp_v2])

        # Runs for v1 (Baseline: Task 1 passed, Task 2 passed, Task 3 failed)
        r_v1_1 = Run(
            task_id="TASK-REG-101",
            agent_name="CopilotAgent",
            agent_version="v1.0",
            model="claude-3-5-haiku",
            experiment_id="exp_copilot_v1",
            dataset_version="1.0",
            latency_ms=1200.0,
            est_cost_usd=0.0012,
            query="Look up order status for #90210",
            created_at=datetime.now(timezone.utc),
        )
        r_v1_2 = Run(
            task_id="TASK-REG-102",
            agent_name="CopilotAgent",
            agent_version="v1.0",
            model="gpt-4o-mini",
            experiment_id="exp_copilot_v1",
            dataset_version="1.0",
            latency_ms=1400.0,
            est_cost_usd=0.0008,
            query="Check stock for SKU-4421",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([r_v1_1, r_v1_2])
        session.commit()

        # Evals for v1
        e_v1_1 = EvalResult(run_id=r_v1_1.id, metric_name="task_success", score=1.0, passed=True, details="Order lookup successful")
        e_v1_2 = EvalResult(run_id=r_v1_2.id, metric_name="task_success", score=1.0, passed=True, details="Stock check successful")
        session.add_all([e_v1_1, e_v1_2])

        # Runs for v2 (Target: Task 1 regressed to failed, Task 2 passed, Task 3 failed)
        r_v2_1 = Run(
            task_id="TASK-REG-101",
            agent_name="CopilotAgent",
            agent_version="v2.0",
            model="claude-3-5-haiku",
            experiment_id="exp_copilot_v2",
            dataset_version="2.0",
            latency_ms=4900.0,
            est_cost_usd=0.0025,
            query="Look up order status for #90210",
            created_at=datetime.now(timezone.utc),
        )
        r_v2_2 = Run(
            task_id="TASK-REG-102",
            agent_name="CopilotAgent",
            agent_version="v2.0",
            model="gpt-4o-mini",
            experiment_id="exp_copilot_v2",
            dataset_version="2.0",
            latency_ms=1100.0,
            est_cost_usd=0.0007,
            query="Check stock for SKU-4421",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([r_v2_1, r_v2_2])
        session.commit()

        # Steps for v2_1 (Tool failures, LLM calls)
        s1 = Step(
            run_id=r_v2_1.id,
            step_index=1,
            step_type="tool_call",
            tool_name="order_db_query",
            latency_ms=800.0,
            status="error",
            error="Connection timeout to order database",
        )
        s2 = Step(
            run_id=r_v2_1.id,
            step_index=2,
            step_type="tool_call",
            tool_name="order_db_query",
            latency_ms=850.0,
            status="error",
            error="Connection timeout retry failed",
        )
        s3 = Step(
            run_id=r_v2_1.id,
            step_index=3,
            step_type="llm_call",
            operation_name="claude_generation",
            latency_ms=2500.0,
            status="success",
        )
        s4 = Step(
            run_id=r_v2_2.id,
            step_index=1,
            step_type="retriever",
            operation_name="rag_knowledge_search",
            latency_ms=300.0,
            status="success",
        )
        session.add_all([s1, s2, s3, s4])

        # Evals for v2
        e_v2_1 = EvalResult(run_id=r_v2_1.id, metric_name="task_success", score=0.0, passed=False, details="Order lookup failed")
        e_v2_2 = EvalResult(run_id=r_v2_1.id, metric_name="tool_selection_accuracy", score=0.0, passed=False, details="Tool order_db_query threw timeout")
        e_v2_3 = EvalResult(run_id=r_v2_1.id, metric_name="latency_budget", score=0.0, passed=False, details="Latency 4900ms breached budget")
        e_v2_4 = EvalResult(run_id=r_v2_2.id, metric_name="task_success", score=1.0, passed=True, details="Stock check succeeded")
        e_v2_5 = EvalResult(run_id=r_v2_2.id, metric_name="keyword_groundedness", score=0.65, passed=False, details="Low grounding on SKU docs")
        session.add_all([e_v2_1, e_v2_2, e_v2_3, e_v2_4, e_v2_5])
        session.commit()

    finally:
        session.close()


def test_copilot_q1_why_agent_failing():
    """Verify Copilot diagnoses why the agent is failing with concrete factual evidence."""
    res = EvaluationCopilot.ask("Why is my agent failing?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert len(res.findings) >= 1
    
    # Check 3-part structured format
    f = res.findings[0]
    assert len(f.finding) > 0
    assert len(f.evidence) > 0
    assert len(f.recommendation) > 0
    assert "failed" in f.evidence.lower() or "failure" in f.evidence.lower()


def test_copilot_q2_which_tools_cause_failures():
    """Verify Copilot identifies tool failure culprits."""
    res = EvaluationCopilot.ask("Which tools cause the most failures?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert "order_db_query" in res.direct_answer or any("order_db_query" in f.evidence for f in res.findings)


def test_copilot_q3_which_test_cases_regress():
    """Verify Copilot tracks regressions across version transitions."""
    # Global regression check
    res = EvaluationCopilot.ask("Which test cases regress most often?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert len(res.findings) >= 1
    assert len(res.supporting_table) >= 1
    assert "regress" in res.findings[0].finding.lower() or "regression" in res.findings[0].evidence.lower()

    # Agent-scoped regression check
    res_agent = EvaluationCopilot.ask("Which test cases regress most often?", agent_name="agent_copilot")
    assert isinstance(res_agent, CopilotResponse)
    assert res_agent.insufficient_data is False
    assert "TASK-REG-101" in res_agent.direct_answer or any("TASK-REG-101" in f.evidence for f in res_agent.findings)


def test_copilot_q4_which_agent_version_is_better():
    """Verify Copilot compares versions objectively based on scores and latency."""
    res = EvaluationCopilot.ask("Which agent version is better?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert len(res.findings) >= 1
    assert "Version `" in res.direct_answer or "superior" in res.direct_answer or "v1.0" in res.direct_answer or "v2.0" in res.direct_answer


def test_copilot_q5_why_latency_increased():
    """Verify Copilot dissects latency by span operation type."""
    res = EvaluationCopilot.ask("Why did latency increase?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert res.intent == "why_latency_increased"
    assert any("span" in f.evidence.lower() or "latency" in f.evidence.lower() or "ms" in f.evidence.lower() or "llm" in f.evidence.lower() or "tool" in f.evidence.lower() for f in res.findings)


def test_copilot_q6_which_model_is_cost_efficient():
    """Verify Copilot evaluates cost-to-quality efficiency across models."""
    res = EvaluationCopilot.ask("Which model is most cost-efficient?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert "gpt-4o-mini" in res.direct_answer or "claude-3-5-haiku" in res.direct_answer


def test_copilot_q7_common_failure_categories():
    """Verify Copilot breaks down distribution across failure classifications."""
    res = EvaluationCopilot.ask("What are the most common failure categories?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert len(res.findings) >= 1


def test_copilot_q8_rag_poor_retrieval():
    """Verify Copilot isolates RAG queries with low context precision or grounding."""
    res = EvaluationCopilot.ask("Which RAG queries have poor retrieval?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert "TASK-REG-102" in res.direct_answer or len(res.findings) >= 1


def test_copilot_q9_unnecessary_tool_calls():
    """Verify Copilot detects redundant repeated tool calls."""
    res = EvaluationCopilot.ask("Which tools are being called unnecessarily?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert "order_db_query" in res.direct_answer or len(res.findings) >= 1


def test_copilot_q10_what_to_investigate_first():
    """Verify Copilot produces prioritized triage action items."""
    res = EvaluationCopilot.ask("What should I investigate first?")
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is False
    assert len(res.findings) >= 1
    priorities = [f.priority for f in res.findings]
    assert any("P0" in p or "P1" in p for p in priorities)


def test_copilot_insufficient_data_fallback():
    """Verify explicit fallback when querying non-existent scope."""
    res = EvaluationCopilot.ask(
        query="Why is my agent failing?",
        agent_name="NonExistentAgent12345",
        experiment_id="exp_non_existent_99999",
    )
    assert isinstance(res, CopilotResponse)
    assert res.insufficient_data is True
    assert "Insufficient evaluation data to determine this." in res.direct_answer
