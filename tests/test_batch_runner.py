"""
Unit and integration tests for BatchEvaluationRunner service.
"""

import os
import sys
import time
import pytest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db, get_session
from src.storage.models import Run
from src.core.test_case_manager import TestCaseManager
from src.core.dataset_manager import DatasetManager
from src.evaluation.batch_runner import BatchEvaluationRunner
from src.agent.demo_agent import DemoAgent


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_batch_evaluation_non_blocking_and_scorecard():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)
    runner = BatchEvaluationRunner()

    # Create 3 test cases
    tc1 = tc_mgr.create_test_case({
        "test_id": f"TC_B_{uuid.uuid4().hex[:6]}",
        "name": "Batch Math",
        "user_input": "Calculate 15% discount on $200",
        "expected_tools": ["calculator"],
        "expected_keywords": ["170"],
    })
    tc2 = tc_mgr.create_test_case({
        "test_id": f"TC_B_{uuid.uuid4().hex[:6]}",
        "name": "Batch Knowledge",
        "user_input": "What is the return policy?",
        "expected_tools": ["search_knowledge_base"],
        "expected_keywords": ["30 days"],
    })
    tc3 = tc_mgr.create_test_case({
        "test_id": f"TC_B_{uuid.uuid4().hex[:6]}",
        "name": "Batch Disabled Test",
        "user_input": "Disabled query",
        "enabled": False,
    })

    ds_id = f"ds_batch_{uuid.uuid4().hex[:6]}"
    dataset = ds_mgr.create_dataset(
        name="Automated Test Dataset",
        version="1.0",
        test_case_ids=[tc1.test_id, tc2.test_id, tc3.test_id],
        dataset_id=ds_id,
    )

    agent = DemoAgent(use_mock=True)

    # 1. Start batch run asynchronously
    job_id = runner.start_batch_run(agent=agent, dataset=dataset, model="claude-3-5-haiku")
    assert job_id is not None
    assert job_id.startswith("batch_")

    # 2. Check that start_batch_run returned immediately (non-blocking)
    job = runner.get_job(job_id)
    assert job is not None
    assert job.status in ["pending", "running", "completed"]

    # 3. Wait for background thread to complete
    max_wait = 15.0
    start = time.time()
    while job.status in ["pending", "running"] and (time.time() - start) < max_wait:
        time.sleep(0.1)

    assert job.status == "completed"
    assert job.total_tests == 3
    assert job.completed_tests == 3
    assert job.skipped_count == 1  # tc3 was disabled
    assert job.progress_pct == 100.0

    # 4. Check evaluation summary scorecard
    summary = runner.get_evaluation_run_summary(job.experiment_id)
    assert summary["experiment_id"] == job.experiment_id
    assert summary["total_tests"] == 2  # 2 runs persisted
    assert "average_latency_ms" in summary
    assert "p95_latency_ms" in summary
    assert "overall_score" in summary
    assert "metric_scores" in summary
    assert "total_cost_usd" in summary
    assert "total_tokens" in summary
    assert len(summary["test_runs"]) == 2

    # Verify individual runs are queryable in SQLite
    session = get_session()
    try:
        db_runs = session.query(Run).filter(Run.experiment_id == job.experiment_id).all()
        assert len(db_runs) == 2
        for r in db_runs:
            assert r.experiment_id == job.experiment_id
            assert len(r.eval_results) > 0
    finally:
        session.close()


def test_batch_evaluation_cancellation():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)
    runner = BatchEvaluationRunner()

    # Create several test cases
    tc_ids = []
    for i in range(10):
        tc = tc_mgr.create_test_case({
            "test_id": f"TC_CANCEL_{i}_{uuid.uuid4().hex[:4]}",
            "name": f"Cancel Test {i}",
            "user_input": f"Query {i}",
        })
        tc_ids.append(tc.test_id)

    ds_id = f"ds_cancel_{uuid.uuid4().hex[:6]}"
    dataset = ds_mgr.create_dataset(
        name="Cancellation Suite",
        version="1.0",
        test_case_ids=tc_ids,
        dataset_id=ds_id,
    )

    agent = DemoAgent(use_mock=True)

    job_id = runner.start_batch_run(agent=agent, dataset=dataset)

    # Immediately request cancellation
    cancelled = runner.cancel_job(job_id)
    assert cancelled is True

    # Wait for completion/cancellation
    time.sleep(1.0)
    job = runner.get_job(job_id)
    assert job.status in ["cancelled", "completed"]
