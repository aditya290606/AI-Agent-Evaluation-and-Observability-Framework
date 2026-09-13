"""
Evaluation Dataset Manager service.

Provides CRUD, versioning, test case membership, duplication, version comparison,
and database persistence for Evaluation Datasets.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd

from src.storage.db import get_session, init_db
from src.storage.models import DatasetRecord, TestCaseRecord, Run, EvalResult
from src.core.entities import EvaluationDataset, TestCase
from src.core.test_case_manager import TestCaseManager


class DatasetManager:
    """Manages evaluation datasets, versions, test case associations, and version diffing."""

    def __init__(self, tc_manager: Optional[TestCaseManager] = None):
        init_db()
        self.tc_manager = tc_manager or TestCaseManager()

    def _record_to_entity(self, record: DatasetRecord) -> EvaluationDataset:
        test_ids = []
        if record.test_case_ids:
            try:
                test_ids = json.loads(record.test_case_ids)
            except Exception:
                test_ids = []

        # Hydrate full TestCase objects
        test_cases: List[TestCase] = []
        for tid in test_ids:
            tc = self.tc_manager.get_test_case(tid)
            if tc:
                test_cases.append(tc)
            else:
                # Placeholder if test case record was removed
                test_cases.append(TestCase(test_id=tid, name=f"Test {tid}", user_input=""))

        meta = {}
        if record.metadata_json:
            try:
                meta = json.loads(record.metadata_json)
            except Exception:
                meta = {}

        return EvaluationDataset(
            name=record.name,
            id=record.dataset_id,
            description=record.description or "",
            version=record.version or "1.0",
            test_cases=test_cases,
            created_at=record.created_at or datetime.now(timezone.utc),
            metadata=meta,
            _dataset_id=record.dataset_id,
        )

    def create_dataset(
        self,
        name: str,
        description: str = "",
        version: str = "1.0",
        test_case_ids: Optional[List[str]] = None,
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationDataset:
        """Create and persist a new evaluation dataset."""
        session = get_session()
        try:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Dataset name cannot be empty.")

            clean_version = (version or "1.0").strip()
            clean_id = (dataset_id or f"ds_{uuid.uuid4().hex[:8]}").strip().lower().replace(" ", "_")

            # Check for existing dataset with same id and version
            existing = session.query(DatasetRecord).filter(
                DatasetRecord.dataset_id == clean_id,
                DatasetRecord.version == clean_version
            ).first()
            if existing:
                raise ValueError(f"Dataset '{clean_id}' version '{clean_version}' already exists.")

            t_ids = list(dict.fromkeys(test_case_ids or []))  # deduplicate preserving order

            record = DatasetRecord(
                dataset_id=clean_id,
                name=clean_name,
                description=description.strip() if description else "",
                version=clean_version,
                test_case_ids=json.dumps(t_ids),
                metadata_json=json.dumps(metadata or {}),
            )
            session.add(record)
            session.commit()
            return self._record_to_entity(record)
        finally:
            session.close()

    def get_dataset(self, dataset_id: str, version: Optional[str] = None) -> Optional[EvaluationDataset]:
        """Fetch an evaluation dataset by ID and optional version (defaults to latest version)."""
        session = get_session()
        try:
            query = session.query(DatasetRecord).filter(DatasetRecord.dataset_id == dataset_id.strip())
            if version:
                record = query.filter(DatasetRecord.version == version.strip()).first()
            else:
                record = query.order_by(DatasetRecord.created_at.desc()).first()

            return self._record_to_entity(record) if record else None
        finally:
            session.close()

    def list_datasets(self, latest_only: bool = True) -> List[EvaluationDataset]:
        """List all datasets. If latest_only is True, returns only the latest version of each dataset."""
        session = get_session()
        try:
            records = session.query(DatasetRecord).order_by(DatasetRecord.name.asc(), DatasetRecord.created_at.desc()).all()
            if not latest_only:
                return [self._record_to_entity(r) for r in records]

            # Keep only latest version per dataset_id
            seen_ids = set()
            latest_entities = []
            for r in records:
                if r.dataset_id not in seen_ids:
                    seen_ids.add(r.dataset_id)
                    latest_entities.append(self._record_to_entity(r))
            return latest_entities
        finally:
            session.close()

    def list_dataset_versions(self, dataset_id: str) -> List[EvaluationDataset]:
        """List all available versions for a given dataset_id, ordered chronologically."""
        session = get_session()
        try:
            records = session.query(DatasetRecord).filter(
                DatasetRecord.dataset_id == dataset_id.strip()
            ).order_by(DatasetRecord.created_at.asc()).all()
            return [self._record_to_entity(r) for r in records]
        finally:
            session.close()

    def add_test_cases(
        self,
        dataset_id: str,
        test_case_ids: List[str],
        version: Optional[str] = None,
    ) -> EvaluationDataset:
        """Add test cases to an existing dataset version."""
        session = get_session()
        try:
            query = session.query(DatasetRecord).filter(DatasetRecord.dataset_id == dataset_id.strip())
            if version:
                record = query.filter(DatasetRecord.version == version.strip()).first()
            else:
                record = query.order_by(DatasetRecord.created_at.desc()).first()

            if not record:
                raise KeyError(f"Dataset '{dataset_id}' (version: {version or 'latest'}) not found.")

            current_ids = json.loads(record.test_case_ids or "[]")
            for tid in test_case_ids:
                if tid not in current_ids:
                    current_ids.append(tid)

            record.test_case_ids = json.dumps(current_ids)
            session.commit()
            return self._record_to_entity(record)
        finally:
            session.close()

    def remove_test_cases(
        self,
        dataset_id: str,
        test_case_ids: List[str],
        version: Optional[str] = None,
    ) -> EvaluationDataset:
        """Remove test cases from an existing dataset version."""
        session = get_session()
        try:
            query = session.query(DatasetRecord).filter(DatasetRecord.dataset_id == dataset_id.strip())
            if version:
                record = query.filter(DatasetRecord.version == version.strip()).first()
            else:
                record = query.order_by(DatasetRecord.created_at.desc()).first()

            if not record:
                raise KeyError(f"Dataset '{dataset_id}' (version: {version or 'latest'}) not found.")

            current_ids = json.loads(record.test_case_ids or "[]")
            remove_set = set(test_case_ids)
            filtered_ids = [tid for tid in current_ids if tid not in remove_set]

            record.test_case_ids = json.dumps(filtered_ids)
            session.commit()
            return self._record_to_entity(record)
        finally:
            session.close()

    def duplicate_dataset(
        self,
        dataset_id: str,
        new_name: str,
        new_dataset_id: Optional[str] = None,
        source_version: Optional[str] = None,
    ) -> EvaluationDataset:
        """Duplicate an existing dataset version into a new dataset."""
        source = self.get_dataset(dataset_id, version=source_version)
        if not source:
            raise KeyError(f"Source dataset '{dataset_id}' (version: {source_version or 'latest'}) not found.")

        target_id = new_dataset_id or f"{source.dataset_id}_copy_{uuid.uuid4().hex[:4]}"
        test_ids = [tc.test_id for tc in source.test_cases]

        return self.create_dataset(
            name=new_name.strip(),
            description=f"Duplicate of {source.name} ({source.version}): {source.description}",
            version="1.0",
            test_case_ids=test_ids,
            dataset_id=target_id,
            metadata=dict(source.metadata),
        )

    def version_dataset(
        self,
        dataset_id: str,
        new_version: str,
        description: Optional[str] = None,
        test_case_ids: Optional[List[str]] = None,
        source_version: Optional[str] = None,
    ) -> EvaluationDataset:
        """Create a new version snapshot for an existing dataset."""
        source = self.get_dataset(dataset_id, version=source_version)
        if not source:
            raise KeyError(f"Source dataset '{dataset_id}' not found.")

        clean_ver = new_version.strip()
        t_ids = test_case_ids if test_case_ids is not None else [tc.test_id for tc in source.test_cases]
        desc = description if description is not None else source.description

        return self.create_dataset(
            name=source.name,
            description=desc,
            version=clean_ver,
            test_case_ids=t_ids,
            dataset_id=source.dataset_id,
            metadata=dict(source.metadata),
        )

    def delete_dataset(self, dataset_id: str, version: Optional[str] = None) -> bool:
        """Delete a specific version or all versions of a dataset."""
        session = get_session()
        try:
            query = session.query(DatasetRecord).filter(DatasetRecord.dataset_id == dataset_id.strip())
            if version:
                query = query.filter(DatasetRecord.version == version.strip())

            records = query.all()
            if not records:
                return False

            for r in records:
                session.delete(r)
            session.commit()
            return True
        finally:
            session.close()

    def compare_dataset_versions(
        self,
        dataset_id: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between two versions of a dataset.

        Returns:
          - version_a, version_b metadata
          - added_test_cases (in B not in A)
          - removed_test_cases (in A not in B)
          - common_test_cases (in both)
          - modified_test_cases (structural/content changes)
          - evaluation_diff (pass rate, quality score, latency, cost delta if runs exist)
        """
        ds_a = self.get_dataset(dataset_id, version=version_a)
        ds_b = self.get_dataset(dataset_id, version=version_b)

        if not ds_a:
            raise KeyError(f"Dataset '{dataset_id}' version '{version_a}' not found.")
        if not ds_b:
            raise KeyError(f"Dataset '{dataset_id}' version '{version_b}' not found.")

        map_a = {tc.test_id: tc for tc in ds_a.test_cases}
        map_b = {tc.test_id: tc for tc in ds_b.test_cases}

        set_a = set(map_a.keys())
        set_b = set(map_b.keys())

        added_ids = sorted(list(set_b - set_a))
        removed_ids = sorted(list(set_a - set_b))
        common_ids = sorted(list(set_a.intersection(set_b)))

        modified_tests = []
        for tid in common_ids:
            tc_a = map_a[tid]
            tc_b = map_b[tid]
            diffs = {}
            if tc_a.user_input != tc_b.user_input:
                diffs["user_input"] = {"from": tc_a.user_input, "to": tc_b.user_input}
            if tc_a.expected_tools != tc_b.expected_tools:
                diffs["expected_tools"] = {"from": tc_a.expected_tools, "to": tc_b.expected_tools}
            if tc_a.expected_answer != tc_b.expected_answer:
                diffs["expected_answer"] = {"from": tc_a.expected_answer, "to": tc_b.expected_answer}
            if tc_a.latency_budget != tc_b.latency_budget:
                diffs["latency_budget"] = {"from": tc_a.latency_budget, "to": tc_b.latency_budget}
            if tc_a.difficulty != tc_b.difficulty:
                diffs["difficulty"] = {"from": tc_a.difficulty, "to": tc_b.difficulty}

            if diffs:
                modified_tests.append({
                    "test_id": tid,
                    "name": tc_b.name,
                    "field_changes": diffs,
                })

        # Check historical runs for both versions to compute benchmark deltas
        eval_diff = self._compare_version_runs(dataset_id, version_a, version_b)

        return {
            "dataset_id": dataset_id,
            "dataset_name": ds_a.name,
            "version_a": {
                "version": version_a,
                "total_tests": len(ds_a.test_cases),
                "created_at": str(ds_a.created_at),
                "description": ds_a.description,
            },
            "version_b": {
                "version": version_b,
                "total_tests": len(ds_b.test_cases),
                "created_at": str(ds_b.created_at),
                "description": ds_b.description,
            },
            "added_test_cases": [map_b[tid].to_dict() for tid in added_ids],
            "removed_test_cases": [map_a[tid].to_dict() for tid in removed_ids],
            "common_test_ids": common_ids,
            "modified_test_cases": modified_tests,
            "evaluation_diff": eval_diff,
        }

    def _compare_version_runs(
        self,
        dataset_id: str,
        version_a: str,
        version_b: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and compare recent evaluation runs for version_a vs version_b if available."""
        session = get_session()
        try:
            # Match runs where dataset_version or experiment metadata matches
            runs_a = session.query(Run).filter(Run.dataset_version == version_a).all()
            runs_b = session.query(Run).filter(Run.dataset_version == version_b).all()

            if not runs_a or not runs_b:
                return None

            def get_run_metrics(runs_list):
                total = len(runs_list)
                total_latency = sum(r.latency_ms or 0 for r in runs_list)
                avg_latency = total_latency / total if total else 0
                total_cost = sum(r.est_cost_usd or 0 for r in runs_list)

                # Collect eval results
                passed_checks = 0
                total_checks = 0
                for r in runs_list:
                    for ev in r.eval_results:
                        total_checks += 1
                        if ev.passed:
                            passed_checks += 1

                pass_rate = (passed_checks / total_checks * 100) if total_checks else 0
                return {
                    "total_runs": total,
                    "pass_rate_pct": round(pass_rate, 1),
                    "avg_latency_ms": round(avg_latency, 1),
                    "total_cost_usd": round(total_cost, 4),
                }

            metrics_a = get_run_metrics(runs_a)
            metrics_b = get_run_metrics(runs_b)

            return {
                "has_benchmark_data": True,
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "pass_rate_delta": round(metrics_b["pass_rate_pct"] - metrics_a["pass_rate_pct"], 1),
                "latency_delta_ms": round(metrics_b["avg_latency_ms"] - metrics_a["avg_latency_ms"], 1),
                "cost_delta_usd": round(metrics_b["total_cost_usd"] - metrics_a["total_cost_usd"], 4),
            }
        finally:
            session.close()

    def seed_default_dataset_if_empty(self) -> Optional[EvaluationDataset]:
        """Seed the baseline golden tasks into the datasets table if no datasets exist."""
        session = get_session()
        try:
            count = session.query(DatasetRecord).count()
            if count > 0:
                return None
        finally:
            session.close()

        # Gather existing test case IDs or baseline IDs
        all_tcs = self.tc_manager.list_test_cases()
        if not all_tcs:
            from src.core.dataset import DatasetLoader
            default_ds = DatasetLoader.load_default_dataset()
            tc_ids = []
            for tc in default_ds.test_cases:
                try:
                    self.tc_manager.create_test_case(tc.to_dict())
                except Exception:
                    pass
                tc_ids.append(tc.test_id)
        else:
            tc_ids = [tc.test_id for tc in all_tcs]

        return self.create_dataset(
            name="Golden Regression Tasks",
            description="Baseline enterprise evaluation tasks covering retrieval, math, reasoning, and tool execution.",
            version="1.0",
            test_case_ids=tc_ids,
            dataset_id="golden_tasks",
            metadata={"system_seeded": True, "category": "golden_benchmark"},
        )
