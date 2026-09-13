"""
Dataset loading and management utilities.

Allows evaluation suites to load test cases from JSON, CSV, or memory,
making the test suite fully configurable rather than hardcoding T001..T015.
"""

import json
import os
from typing import List, Union, Dict, Any, Optional
from src.core.entities import EvaluationDataset, TestCase

DEFAULT_GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "golden_tasks.json",
)


class DatasetLoader:
    """Helper for loading and creating EvaluationDatasets."""

    @staticmethod
    def from_json(filepath: str, name: Optional[str] = None, description: str = "") -> EvaluationDataset:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evaluation dataset file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Dataset file must contain a list of test case objects, got {type(raw_data)}")

        dataset_name = name or os.path.splitext(os.path.basename(filepath))[0]
        dataset = EvaluationDataset(name=dataset_name, description=description)

        for item in raw_data:
            case = TestCase.from_dict(item)
            dataset.add_test_case(case)

        return dataset

    @staticmethod
    def from_dict_list(data: List[Dict[str, Any]], name: str = "custom_dataset", description: str = "") -> EvaluationDataset:
        dataset = EvaluationDataset(name=name, description=description)
        for item in data:
            dataset.add_test_case(TestCase.from_dict(item))
        return dataset

    @classmethod
    def load_default_dataset(cls) -> EvaluationDataset:
        """Loads the framework's baseline 15 golden evaluation tasks."""
        return cls.from_json(DEFAULT_GOLDEN_PATH, name="golden_tasks", description="Baseline 15 regression tasks")

    @classmethod
    def from_database(
        cls,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
        name: str = "database_test_cases",
        enabled_only: bool = True,
    ) -> EvaluationDataset:
        """Loads test cases directly from the persistent database."""
        from src.core.test_case_manager import TestCaseManager
        mgr = TestCaseManager()
        return mgr.to_evaluation_dataset(tag=tag, difficulty=difficulty, dataset_name=name, enabled_only=enabled_only)
