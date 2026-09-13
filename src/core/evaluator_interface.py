"""
Evaluator interface contract.

All evaluators (deterministic, semantic, model-based, budget, structural, behavioral, reliability)
implement BaseEvaluator with the standard signature:
    evaluate(test_case, execution_result, trace) -> EvaluationResult

Maintains seamless backward compatibility with the legacy evaluate(trace, test_case) signature.
"""

from abc import ABC, abstractmethod
import time
from typing import Optional, Any
from src.core.entities import Trace, TestCase, EvaluationResult


class BaseEvaluator(ABC):
    """Abstract interface for modular evaluators."""

    def __init__(
        self,
        name: str,
        threshold: float = 1.0,
        weight: float = 1.0,
        evaluator_type: str = "deterministic",
        description: str = "",
    ):
        self.name = name
        self.threshold = threshold
        self.weight = weight
        self.evaluator_type = evaluator_type
        self.description = description

    @abstractmethod
    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        """
        Evaluate an agent's execution against a test case.
        """
        pass

    def __call__(self, *args, **kwargs) -> EvaluationResult:
        """
        Flexible caller supporting:
        1. evaluate(test_case, execution_result, trace)
        2. evaluate(test_case, trace)
        3. evaluate(trace, test_case) [legacy BaseMetric compatibility]
        """
        start = time.time()

        test_case = None
        execution_result = kwargs.get("execution_result")
        trace = kwargs.get("trace")

        if len(args) == 3:
            test_case, execution_result, trace = args
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, Trace) and isinstance(arg1, TestCase):
                # Legacy: (trace, test_case)
                trace, test_case = arg0, arg1
                execution_result = getattr(trace, "final_answer", None)
            elif isinstance(arg0, TestCase) and isinstance(arg1, Trace):
                # (test_case, trace)
                test_case, trace = arg0, arg1
                execution_result = getattr(trace, "final_answer", None)
            elif isinstance(arg0, TestCase):
                test_case = arg0
                execution_result = arg1
            else:
                test_case, trace = arg0, arg1
        elif len(args) == 1:
            if isinstance(args[0], TestCase):
                test_case = args[0]
            elif isinstance(args[0], Trace):
                trace = args[0]
                test_case = kwargs.get("test_case")

        # Fallback keyword extraction
        if test_case is None and "test_case" in kwargs:
            test_case = kwargs["test_case"]
        if trace is None and "trace" in kwargs:
            trace = kwargs["trace"]

        try:
            result = self.evaluate(test_case=test_case, execution_result=execution_result, trace=trace)
        except TypeError:
            try:
                # Legacy evaluate(trace, test_case)
                result = self.evaluate(trace, test_case)
            except TypeError:
                try:
                    # Legacy evaluate(test_case, trace)
                    result = self.evaluate(test_case, trace)
                except TypeError:
                    result = self.evaluate(test_case)

        result.execution_time_ms = (time.time() - start) * 1000
        if not getattr(result, "threshold", None):
            result.threshold = self.threshold
        if not getattr(result, "evaluator_type", None):
            result.evaluator_type = self.evaluator_type

        # Never present mock scores as real evaluation results
        if trace and getattr(trace, "is_mock", False):
            result.is_mock = True
            result.evaluation_type = "mock"
            if not result.explanation.startswith("[MOCK EVALUATION]"):
                result.explanation = f"[MOCK EVALUATION] {result.explanation}"
                result.details = result.explanation

        return result


# Core aliases
Evaluator = BaseEvaluator
BaseMetric = BaseEvaluator
