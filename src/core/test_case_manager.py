"""
Test Case Manager service.

Provides CRUD, cloning, enabling/disabling, tag filtering, single-test execution,
and dataset conversion for Evaluation Test Cases.
"""

import json
import os
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple, Callable

from src.storage.db import get_session, init_db
from src.storage.models import TestCaseRecord
from src.core.entities import TestCase, EvaluationDataset, Trace, EvaluationResult
from src.core.agent_interface import BaseAgent
from src.core.metric_interface import BaseMetric
from src.evaluation.metrics import (
    ExactAnswerMetric,
    SemanticAnswerMetric,
    ToolSelectionMetric,
    ToolArgumentMetric,
    ExpectedBehaviorMetric,
    GroundednessMetric,
    LatencyMetric,
    StructuredOutputMetric,
    SafetyConstraintMetric,
)


class TestCaseManager:
    """Manages the lifecycle and execution of evaluation test cases."""
    __test__ = False

    METRIC_REGISTRY = {
        "exact_answer": ExactAnswerMetric,
        "semantic_answer": SemanticAnswerMetric,
        "tool_selection": ToolSelectionMetric,
        "tool_arguments": ToolArgumentMetric,
        "expected_behavior": ExpectedBehaviorMetric,
        "groundedness": GroundednessMetric,
        "keyword_groundedness": GroundednessMetric,
        "latency": LatencyMetric,
        "structured_output": StructuredOutputMetric,
        "safety_constraints": SafetyConstraintMetric,
    }

    @staticmethod
    def _record_to_entity(r: TestCaseRecord) -> TestCase:
        return TestCase(
            test_id=r.test_id,
            name=r.name,
            user_input=r.user_input,
            description=r.description or "",
            expected_behavior=r.expected_behavior or "",
            expected_answer=r.expected_answer,
            expected_tools=json.loads(r.expected_tools or "[]"),
            forbidden_tools=json.loads(r.forbidden_tools or "[]"),
            expected_keywords=json.loads(r.expected_keywords or "[]"),
            latency_budget=float(r.latency_budget or 5000.0),
            expected_output_schema=json.loads(r.expected_output_schema) if r.expected_output_schema else None,
            evaluation_metrics=json.loads(r.evaluation_metrics or "[]"),
            tags=json.loads(r.tags or "[]"),
            difficulty=r.difficulty or "medium",
            enabled=bool(r.enabled),
        )

    def __init__(self):
        init_db()
        # Guarantee baseline test cases from golden_tasks.json exist in database
        self.sync_baseline_tests()

    def create_test_case(self, data: Dict[str, Any]) -> TestCase:
        """Create and persist a new evaluation test case with standard fields."""
        session = get_session()
        try:
            test_id = str(data.get("test_id") or data.get("task_id") or f"TC_{uuid.uuid4().hex[:6].upper()}").strip()
            existing = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == test_id).first()
            if existing:
                raise ValueError(f"Test case with ID '{test_id}' already exists.")

            tool = data.get("expected_tool")
            tools = [tool] if tool else (data.get("expected_tools") or [])
            forb_tools = data.get("forbidden_tools") or []
            
            keywords = data.get("expected_keywords") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            
            expected_answer = data.get("expected_answer")
            if not keywords and expected_answer:
                keywords = [k.strip() for k in expected_answer.split(",") if k.strip()]

            metrics = data.get("evaluation_metrics") or ["tool_selection", "keyword_groundedness", "latency", "groundedness"]
            tags = data.get("tags") or []
            schema = data.get("expected_output_schema")

            record = TestCaseRecord(
                test_id=test_id,
                name=data.get("name") or f"Test {test_id}",
                user_input=data.get("user_input") or data.get("query") or "",
                description=data.get("description", ""),
                expected_behavior=data.get("expected_behavior", ""),
                expected_answer=expected_answer,
                expected_tools=json.dumps(tools),
                forbidden_tools=json.dumps(forb_tools),
                expected_keywords=json.dumps(keywords),
                latency_budget=float(data.get("latency_budget", 5000.0)),
                expected_output_schema=json.dumps(schema) if isinstance(schema, dict) else schema,
                evaluation_metrics=json.dumps(metrics),
                tags=json.dumps(tags),
                difficulty=data.get("difficulty", "medium"),
                enabled=bool(data.get("enabled", True)),
            )
            session.add(record)
            session.commit()
            return self._record_to_entity(record)
        finally:
            session.close()

    def update_test_case(self, test_id: str, updates: Dict[str, Any]) -> TestCase:
        """Update an existing test case."""
        session = get_session()
        try:
            record = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == test_id).first()
            if not record:
                raise KeyError(f"Test case with ID '{test_id}' not found.")

            if "name" in updates:
                record.name = updates["name"]
            if "user_input" in updates or "query" in updates:
                record.user_input = updates.get("user_input") or updates.get("query")
            if "description" in updates:
                record.description = updates["description"]
            if "expected_behavior" in updates:
                record.expected_behavior = updates["expected_behavior"]
            if "expected_answer" in updates:
                record.expected_answer = updates["expected_answer"]
            if "expected_tool" in updates:
                tool_val = updates["expected_tool"]
                record.expected_tools = json.dumps([tool_val] if tool_val else [])
            elif "expected_tools" in updates:
                record.expected_tools = json.dumps(updates["expected_tools"])
            if "forbidden_tools" in updates:
                record.forbidden_tools = json.dumps(updates["forbidden_tools"])
            if "expected_keywords" in updates:
                kws = updates["expected_keywords"]
                if isinstance(kws, str):
                    kws = [k.strip() for k in kws.split(",") if k.strip()]
                record.expected_keywords = json.dumps(kws)
            if "latency_budget" in updates:
                record.latency_budget = float(updates["latency_budget"])
            if "expected_output_schema" in updates:
                schema = updates["expected_output_schema"]
                record.expected_output_schema = json.dumps(schema) if isinstance(schema, dict) else schema
            if "evaluation_metrics" in updates:
                record.evaluation_metrics = json.dumps(updates["evaluation_metrics"])
            if "tags" in updates:
                record.tags = json.dumps(updates["tags"])
            if "difficulty" in updates:
                record.difficulty = updates["difficulty"]
            if "enabled" in updates:
                record.enabled = bool(updates["enabled"])

            session.commit()
            return self._record_to_entity(record)
        finally:
            session.close()

    def duplicate_test_case(self, test_id: str, new_id: Optional[str] = None) -> TestCase:
        """Duplicate an existing test case."""
        original = self.get_test_case(test_id)
        if not original:
            raise KeyError(f"Source test case '{test_id}' not found.")

        gen_id = new_id or f"{original.test_id}_copy_{uuid.uuid4().hex[:4]}"
        data = original.to_dict()
        data["test_id"] = gen_id
        data["name"] = f"{original.name} (Copy)"
        return self.create_test_case(data)

    def delete_test_case(self, test_id: str) -> bool:
        """Delete a test case by ID."""
        session = get_session()
        try:
            record = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == test_id).first()
            if not record:
                return False
            session.delete(record)
            session.commit()
            return True
        finally:
            session.close()

    def toggle_test_case(self, test_id: str, enabled: Optional[bool] = None) -> bool:
        """Toggle or explicitly set the enabled status of a test case."""
        session = get_session()
        try:
            record = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == test_id).first()
            if not record:
                raise KeyError(f"Test case '{test_id}' not found.")
            record.enabled = (not record.enabled) if enabled is None else bool(enabled)
            session.commit()
            return record.enabled
        finally:
            session.close()

    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """Fetch a single test case by its ID."""
        session = get_session()
        try:
            record = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == test_id).first()
            return self._record_to_entity(record) if record else None
        finally:
            session.close()

    def list_test_cases(
        self,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
        enabled_only: bool = False,
        search: Optional[str] = None,
    ) -> List[TestCase]:
        """List test cases with optional filtering."""
        session = get_session()
        try:
            query = session.query(TestCaseRecord).order_by(TestCaseRecord.test_id.asc())
            if enabled_only:
                query = query.filter(TestCaseRecord.enabled == True)
            if difficulty and difficulty != "All":
                query = query.filter(TestCaseRecord.difficulty == difficulty.lower())

            records = query.all()
            cases = [self._record_to_entity(r) for r in records]

            if tag and tag != "All":
                cases = [c for c in cases if tag in c.tags]

            if search:
                s_lower = search.lower()
                cases = [
                    c for c in cases
                    if s_lower in c.test_id.lower()
                    or s_lower in c.name.lower()
                    or s_lower in c.user_input.lower()
                ]

            return cases
        finally:
            session.close()

    def get_all_tags(self) -> List[str]:
        """Return a sorted list of all unique tags present across test cases."""
        all_cases = self.list_test_cases()
        tags = set()
        for c in all_cases:
            tags.update(c.tags)
        return sorted(list(tags))

    def run_single_test(
        self,
        agent: BaseAgent,
        test_id: str,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> Tuple[Trace, List[EvaluationResult]]:
        """Execute an agent against a single test case and evaluate it."""
        test_case = self.get_test_case(test_id)
        if not test_case:
            raise KeyError(f"Test case '{test_id}' not found.")

        trace = agent.run(test_case)

        # Resolve metrics
        metric_instances: List[BaseMetric] = []
        if metrics:
            metric_instances = metrics
        else:
            for m_name in test_case.evaluation_metrics:
                cls = self.METRIC_REGISTRY.get(m_name)
                if cls:
                    metric_instances.append(cls())

            # Always ensure latency and tool selection are checked if not present
            if not any(m.name in ["latency", "latency_budget"] for m in metric_instances):
                metric_instances.append(LatencyMetric())
            if test_case.expected_tools and not any(m.name in ["tool_selection", "tool_accuracy"] for m in metric_instances):
                metric_instances.append(ToolSelectionMetric())

        eval_results: List[EvaluationResult] = []
        for metric in metric_instances:
            res = metric.evaluate(trace, test_case)
            eval_results.append(res)

        return trace, eval_results

    def to_evaluation_dataset(
        self,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
        dataset_name: str = "custom_test_set",
        enabled_only: bool = True,
    ) -> EvaluationDataset:
        """Convert filtered test cases into an EvaluationDataset."""
        cases = self.list_test_cases(tag=tag, difficulty=difficulty, enabled_only=enabled_only)
        return EvaluationDataset(
            name=dataset_name,
            description=f"Generated dataset with {len(cases)} test cases (tag={tag or 'all'})",
            test_cases=cases,
        )

    def run_all_tests(
        self,
        agent: BaseAgent,
        enabled_only: bool = True,
        metrics: Optional[List[BaseMetric]] = None,
        progress_callback: Optional[Callable] = None,
        experiment_name: Optional[str] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Execute all (or all enabled) test cases as a test suite against an agent.
        Leverages EvaluationEngine to compute scores, log Runs/Steps/EvalResults,
        and generate a comprehensive EvaluationReport.
        """
        from src.core.engine import EvaluationEngine
        dataset = self.to_evaluation_dataset(enabled_only=enabled_only, dataset_name="configured_suite")
        if not dataset.test_cases:
            raise ValueError("No test cases available to run (check that at least one test is enabled).")

        engine = EvaluationEngine(
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            experiment_name=experiment_name or f"suite_run_{int(time.time())}",
            persist=True,
        )
        report = engine.run(progress_callback=progress_callback)

        test_map = {tc.task_id: tc for tc in dataset.test_cases}
        summaries: List[Dict[str, Any]] = []
        for cs in report.case_results:
            tid = cs.get("test_id", "")
            tc = test_map.get(tid)
            summaries.append({
                "test_id": tid,
                "name": tc.name if tc else f"Test {tid}",
                "user_input": tc.user_input if tc else "",
                "expected_behavior": tc.expected_behavior if tc else "",
                "expected_tool": tc.expected_tool if tc else None,
                "latency_budget": tc.latency_budget if tc else 5000.0,
                "passed": bool(cs.get("passed", False)),
                "score": float(cs.get("weighted_score", 0.0)),
                "failure_reasons": cs.get("failure_reasons", []),
                "individual_scores": cs.get("individual_scores", {}),
            })

        return report, summaries

    def sync_baseline_tests(self, force_overwrite: bool = False) -> int:
        """
        Seed or synchronize baseline golden tasks (T001..T015) from golden_tasks.json into database.
        Ensures all 15 baseline test cases are preserved and have standard metadata.
        """
        golden_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset",
            "golden_tasks.json",
        )
        if not os.path.exists(golden_path):
            return 0

        with open(golden_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

        session = get_session()
        synced_count = 0
        try:
            for task in raw_tasks:
                tid = task.get("test_id") or task.get("task_id", "")
                existing = session.query(TestCaseRecord).filter(TestCaseRecord.test_id == tid).first()

                name = task.get("name") or f"Task {tid}"
                user_input = task.get("user_input") or task.get("query", "")
                expected_behavior = task.get("expected_behavior", "")
                tool = task.get("expected_tool")
                tools = [tool] if tool else task.get("expected_tools", [])
                kws = task.get("expected_keywords") or []
                latency_budget = float(task.get("latency_budget", 5000.0))
                enabled = bool(task.get("enabled", True))

                if not existing:
                    rec = TestCaseRecord(
                        test_id=tid,
                        name=name,
                        user_input=user_input,
                        description=task.get("description", ""),
                        expected_behavior=expected_behavior,
                        expected_answer=task.get("expected_answer"),
                        expected_tools=json.dumps(tools),
                        expected_keywords=json.dumps(kws),
                        latency_budget=latency_budget,
                        enabled=enabled,
                    )
                    session.add(rec)
                    synced_count += 1
                elif force_overwrite or not existing.expected_behavior or existing.expected_behavior.startswith("Should select"):
                    existing.name = name
                    existing.user_input = user_input
                    existing.expected_behavior = expected_behavior
                    existing.expected_tools = json.dumps(tools)
                    existing.expected_keywords = json.dumps(kws)
                    existing.latency_budget = latency_budget
                    synced_count += 1

            session.commit()
            return synced_count
        finally:
            session.close()
