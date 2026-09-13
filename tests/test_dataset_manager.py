"""
Unit and integration tests for DatasetManager service.
"""

import os
import sys
import pytest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db
from src.core.test_case_manager import TestCaseManager
from src.core.dataset_manager import DatasetManager
from src.core.entities import TestCase


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_create_and_get_dataset():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)

    # Create 2 test cases
    tc1 = tc_mgr.create_test_case({"test_id": f"TC_DS_{uuid.uuid4().hex[:6]}", "name": "DS Test 1", "user_input": "Q1"})
    tc2 = tc_mgr.create_test_case({"test_id": f"TC_DS_{uuid.uuid4().hex[:6]}", "name": "DS Test 2", "user_input": "Q2"})

    ds_id = f"ds_test_{uuid.uuid4().hex[:6]}"
    dataset = ds_mgr.create_dataset(
        name="Finance Benchmark",
        description="Core financial question answering benchmark suite",
        version="1.0",
        test_case_ids=[tc1.test_id, tc2.test_id],
        dataset_id=ds_id,
    )

    assert dataset.dataset_id == ds_id
    assert dataset.name == "Finance Benchmark"
    assert dataset.version == "1.0"
    assert len(dataset.test_cases) == 2
    assert dataset.test_cases[0].test_id == tc1.test_id

    # Fetch
    fetched = ds_mgr.get_dataset(ds_id, version="1.0")
    assert fetched is not None
    assert fetched.name == "Finance Benchmark"
    assert len(fetched.test_cases) == 2


def test_add_and_remove_test_cases():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)

    tc1 = tc_mgr.create_test_case({"test_id": f"TC_ADD_{uuid.uuid4().hex[:6]}", "name": "Test A", "user_input": "QA"})
    tc2 = tc_mgr.create_test_case({"test_id": f"TC_ADD_{uuid.uuid4().hex[:6]}", "name": "Test B", "user_input": "QB"})
    tc3 = tc_mgr.create_test_case({"test_id": f"TC_ADD_{uuid.uuid4().hex[:6]}", "name": "Test C", "user_input": "QC"})

    ds_id = f"ds_membership_{uuid.uuid4().hex[:6]}"
    ds_mgr.create_dataset(
        name="Membership Suite",
        version="1.0",
        test_case_ids=[tc1.test_id],
        dataset_id=ds_id,
    )

    # Add test cases
    updated = ds_mgr.add_test_cases(ds_id, [tc2.test_id, tc3.test_id], version="1.0")
    assert len(updated.test_cases) == 3
    assert any(c.test_id == tc2.test_id for c in updated.test_cases)

    # Remove test case
    after_removal = ds_mgr.remove_test_cases(ds_id, [tc2.test_id], version="1.0")
    assert len(after_removal.test_cases) == 2
    assert not any(c.test_id == tc2.test_id for c in after_removal.test_cases)
    assert any(c.test_id == tc1.test_id for c in after_removal.test_cases)
    assert any(c.test_id == tc3.test_id for c in after_removal.test_cases)


def test_duplicate_dataset():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)

    tc1 = tc_mgr.create_test_case({"test_id": f"TC_DUP_{uuid.uuid4().hex[:6]}", "name": "Dup Test", "user_input": "QDup"})
    ds_id = f"ds_orig_{uuid.uuid4().hex[:6]}"

    ds_mgr.create_dataset(
        name="Original Suite",
        description="Original description",
        version="1.0",
        test_case_ids=[tc1.test_id],
        dataset_id=ds_id,
    )

    dup = ds_mgr.duplicate_dataset(
        dataset_id=ds_id,
        new_name="Cloned Suite",
    )

    assert dup.name == "Cloned Suite"
    assert dup.dataset_id != ds_id
    assert len(dup.test_cases) == 1
    assert dup.test_cases[0].test_id == tc1.test_id


def test_version_dataset():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)

    tc1 = tc_mgr.create_test_case({"test_id": f"TC_VER_{uuid.uuid4().hex[:6]}", "name": "V1 Test", "user_input": "Q1"})
    tc2 = tc_mgr.create_test_case({"test_id": f"TC_VER_{uuid.uuid4().hex[:6]}", "name": "V2 Test", "user_input": "Q2"})

    ds_id = f"ds_versioned_{uuid.uuid4().hex[:6]}"
    ds_mgr.create_dataset(
        name="Multi-Version Suite",
        version="1.0",
        test_case_ids=[tc1.test_id],
        dataset_id=ds_id,
    )

    # Create version 2.0 with an extra test case
    v2 = ds_mgr.version_dataset(
        dataset_id=ds_id,
        new_version="2.0",
        description="Added second test case",
        test_case_ids=[tc1.test_id, tc2.test_id],
    )

    assert v2.dataset_id == ds_id
    assert v2.version == "2.0"
    assert len(v2.test_cases) == 2

    # Verify both versions exist
    versions = ds_mgr.list_dataset_versions(ds_id)
    assert len(versions) == 2
    v_tags = [v.version for v in versions]
    assert "1.0" in v_tags
    assert "2.0" in v_tags


def test_compare_dataset_versions():
    tc_mgr = TestCaseManager()
    ds_mgr = DatasetManager(tc_manager=tc_mgr)

    tc_common = tc_mgr.create_test_case({
        "test_id": f"TC_COM_{uuid.uuid4().hex[:6]}",
        "name": "Common Test",
        "user_input": "Common query",
        "difficulty": "easy",
    })
    tc_removed = tc_mgr.create_test_case({
        "test_id": f"TC_REM_{uuid.uuid4().hex[:6]}",
        "name": "To Remove",
        "user_input": "Removed query",
    })
    tc_added = tc_mgr.create_test_case({
        "test_id": f"TC_ADD_{uuid.uuid4().hex[:6]}",
        "name": "Newly Added",
        "user_input": "Added query",
    })

    ds_id = f"ds_diff_{uuid.uuid4().hex[:6]}"

    # Version 1.0 has common + removed
    ds_mgr.create_dataset(
        name="Evolution Dataset",
        version="1.0",
        test_case_ids=[tc_common.test_id, tc_removed.test_id],
        dataset_id=ds_id,
    )

    # Version 2.0 has common + added
    ds_mgr.version_dataset(
        dataset_id=ds_id,
        new_version="2.0",
        test_case_ids=[tc_common.test_id, tc_added.test_id],
    )

    diff = ds_mgr.compare_dataset_versions(ds_id, "1.0", "2.0")

    assert diff["dataset_id"] == ds_id
    assert len(diff["added_test_cases"]) == 1
    assert diff["added_test_cases"][0]["test_id"] == tc_added.test_id

    assert len(diff["removed_test_cases"]) == 1
    assert diff["removed_test_cases"][0]["test_id"] == tc_removed.test_id

    assert tc_common.test_id in diff["common_test_ids"]
