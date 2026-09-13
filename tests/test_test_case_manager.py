"""
Unit tests for TestCaseManager service.
"""

import os
import sys
import pytest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db
from src.core.entities import TestCase
from src.core.test_case_manager import TestCaseManager
from src.agent.demo_agent import DemoAgent


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_test_case_backward_compatibility():
    # Verify task_id, query, expected_tool aliases work identically to test_id, user_input, expected_tools
    tc = TestCase(
        test_id="T_LEGACY_01",
        user_input="How do I reset my password?",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["reset", "email"],
    )

    assert tc.task_id == "T_LEGACY_01"
    assert tc.query == "How do I reset my password?"
    assert tc.expected_tool == "search_knowledge_base"

    # Test alias setters
    tc.task_id = "T_MODIFIED"
    tc.query = "What is the policy?"
    tc.expected_tool = "calculator"

    assert tc.test_id == "T_MODIFIED"
    assert tc.user_input == "What is the policy?"
    assert tc.expected_tools == ["calculator"]


def test_create_and_get_test_case():
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"
    tc = mgr.create_test_case({
        "test_id": uid,
        "name": "Custom Shipping Test",
        "user_input": "How much for overnight shipping?",
        "expected_tools": ["search_knowledge_base"],
        "forbidden_tools": ["execute_sql"],
        "expected_keywords": ["overnight"],
        "latency_budget": 3000.0,
        "tags": ["shipping", "logistics"],
        "difficulty": "easy",
        "enabled": True,
    })

    assert tc.test_id == uid
    assert tc.name == "Custom Shipping Test"
    assert tc.tags == ["shipping", "logistics"]

    fetched = mgr.get_test_case(uid)
    assert fetched is not None
    assert fetched.test_id == uid
    assert fetched.latency_budget == 3000.0


def test_update_test_case():
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"
    mgr.create_test_case({"test_id": uid, "name": "Initial Name", "user_input": "Initial query"})

    updated = mgr.update_test_case(uid, {
        "name": "Updated Name",
        "latency_budget": 1200.0,
        "tags": ["updated_tag"],
    })

    assert updated.name == "Updated Name"
    assert updated.latency_budget == 1200.0
    assert "updated_tag" in updated.tags


def test_duplicate_test_case():
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"
    mgr.create_test_case({"test_id": uid, "name": "To Clone", "user_input": "Clone me", "tags": ["copyable"]})

    cloned = mgr.duplicate_test_case(uid)
    assert cloned.test_id != uid
    assert f"{uid}_copy" in cloned.test_id
    assert "To Clone" in cloned.name
    assert "copyable" in cloned.tags


def test_toggle_and_delete_test_case():
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"
    mgr.create_test_case({"test_id": uid, "name": "Toggle Test", "user_input": "Toggle query", "enabled": True})

    # Disable
    new_state = mgr.toggle_test_case(uid)
    assert new_state is False
    assert mgr.get_test_case(uid).enabled is False

    # Enable
    new_state = mgr.toggle_test_case(uid)
    assert new_state is True
    assert mgr.get_test_case(uid).enabled is True

    # Delete
    assert mgr.delete_test_case(uid) is True
    assert mgr.get_test_case(uid) is None


def test_filter_by_tags_and_dataset_conversion():
    mgr = TestCaseManager()
    tag_name = f"tag_{uuid.uuid4().hex[:4]}"
    mgr.create_test_case({
        "test_id": f"TC_{uuid.uuid4().hex[:6].upper()}",
        "name": "Tagged 1",
        "user_input": "Query 1",
        "tags": [tag_name],
    })
    mgr.create_test_case({
        "test_id": f"TC_{uuid.uuid4().hex[:6].upper()}",
        "name": "Tagged 2",
        "user_input": "Query 2",
        "tags": [tag_name],
    })

    filtered = mgr.list_test_cases(tag=tag_name)
    assert len(filtered) >= 2

    # Dataset conversion
    dataset = mgr.to_evaluation_dataset(tag=tag_name, dataset_name="tagged_dataset")
    assert len(dataset.test_cases) >= 2
    assert dataset.name == "tagged_dataset"


def test_run_single_test():
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"
    mgr.create_test_case({
        "test_id": uid,
        "name": "Math Test",
        "user_input": "What is 245 multiplied by 18?",
        "expected_tools": ["calculator"],
        "expected_keywords": ["4410"],
    })

    agent = DemoAgent(use_mock=True)
    trace, results = mgr.run_single_test(agent, uid)

    assert "4410" in trace.final_answer
    assert len(results) >= 2
    tool_check = next((r for r in results if r.metric_name in ["tool_selection", "tool_accuracy"]), None)
    assert tool_check is not None
    assert tool_check.passed is True


def test_configurable_8_field_schema_lifecycle():
    """Verify test system works with precisely the 8 specified fields:
    test_id, name, user_input, expected_behavior, expected_tool,
    expected_answer/keywords, latency_budget, enabled.
    """
    mgr = TestCaseManager()
    uid = f"TC_{uuid.uuid4().hex[:6].upper()}"

    # 1. Create with 8 fields
    tc = mgr.create_test_case({
        "test_id": uid,
        "name": "Refund Policy Window Test",
        "user_input": "How many days does it take to get a refund after a return is received?",
        "expected_behavior": "Query knowledge base and retrieve 7 business days",
        "expected_tool": "search_knowledge_base",
        "expected_keywords": "7, business days",
        "latency_budget": 4500.0,
        "enabled": True,
    })

    assert tc.test_id == uid
    assert tc.name == "Refund Policy Window Test"
    assert tc.user_input == "How many days does it take to get a refund after a return is received?"
    assert tc.expected_behavior == "Query knowledge base and retrieve 7 business days"
    assert tc.expected_tool == "search_knowledge_base"
    assert "7" in tc.expected_keywords
    assert "business days" in tc.expected_keywords
    assert tc.latency_budget == 4500.0
    assert tc.enabled is True

    # 2. Edit
    updated = mgr.update_test_case(uid, {
        "name": "Updated Refund Policy Test",
        "expected_tool": "search_knowledge_base",
        "expected_keywords": ["7", "business days", "refund"],
        "latency_budget": 4000.0,
    })
    assert updated.name == "Updated Refund Policy Test"
    assert updated.latency_budget == 4000.0
    assert len(updated.expected_keywords) == 3

    # 3. Enable / Disable
    assert mgr.toggle_test_case(uid, False) is False
    assert mgr.get_test_case(uid).enabled is False
    assert mgr.toggle_test_case(uid, True) is True
    assert mgr.get_test_case(uid).enabled is True

    # 4. Run Single Test
    agent = DemoAgent(use_mock=True)
    trace, results = mgr.run_single_test(agent, uid)
    assert trace is not None
    assert len(results) >= 2


def test_run_all_tests_suite_execution():
    """Verify run_all_tests executes the suite of enabled test cases."""
    mgr = TestCaseManager()
    agent = DemoAgent(use_mock=True)

    report, summaries = mgr.run_all_tests(agent, enabled_only=True)
    assert report is not None
    assert report.total_test_cases > 0
    assert report.overall_pass_rate >= 0.0
    assert len(summaries) == report.total_test_cases

    first = summaries[0]
    assert "test_id" in first
    assert "name" in first
    assert "passed" in first
    assert "score" in first


def test_baseline_preservation_t001_to_t015():
    """Verify all 15 baseline test cases (T001..T015) are preserved and configurable."""
    mgr = TestCaseManager()
    mgr.sync_baseline_tests(force_overwrite=True)

    for i in range(1, 16):
        tid = f"T{i:03d}"
        tc = mgr.get_test_case(tid)
        assert tc is not None, f"Baseline test case {tid} must be preserved"
        assert tc.name, f"Test {tid} must have a non-empty name"
        assert tc.user_input, f"Test {tid} must have a non-empty user_input"
        assert tc.expected_behavior, f"Test {tid} must have an expected_behavior"
        assert tc.latency_budget > 0
        assert tc.enabled is True

