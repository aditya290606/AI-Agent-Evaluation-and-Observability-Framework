"""
Analysis and diagnostic modules for agent evaluation and observability.
"""

from src.analysis.root_cause import (
    FailureCategory,
    ObservedFacts,
    RootCauseDiagnosis,
    RootCauseAnalyzer,
)
from src.analysis.regression import (
    Direction,
    TransitionType,
    MetricComparison,
    TestCaseDiff,
    QualityGatePolicy,
    VersionComparisonReport,
    VersionComparator,
)

from src.analysis.report_generator import (
    ExecutiveSummary,
    EvaluationReportData,
    EvaluationReportGenerator,
)
from src.analysis.copilot import (
    CopilotFinding,
    CopilotResponse,
    EvaluationCopilot,
)

__all__ = [
    "FailureCategory",
    "ObservedFacts",
    "RootCauseDiagnosis",
    "RootCauseAnalyzer",
    "Direction",
    "TransitionType",
    "MetricComparison",
    "TestCaseDiff",
    "QualityGatePolicy",
    "VersionComparisonReport",
    "VersionComparator",
    "ExecutiveSummary",
    "EvaluationReportData",
    "EvaluationReportGenerator",
    "CopilotFinding",
    "CopilotResponse",
    "EvaluationCopilot",
]

