"""
Core domain entities for the Agent Evaluation & Observability Framework.

Entities:
  - Agent & AgentVersion
  - EvaluationDataset & TestCase
  - Span & Trace
  - EvaluationResult & Metric
  - Experiment & EvaluationReport
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
import uuid
import time


def _generate_id() -> str:
    return str(uuid.uuid4())[:8]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Agent & AgentVersion
# ---------------------------------------------------------------------------

@dataclass
class AgentVersion:
    version_tag: str
    id: str = field(default_factory=_generate_id)
    agent_id: str = ""
    model_name: str = ""
    prompt_template: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class Agent:
    name: str
    id: str = field(default_factory=_generate_id)
    description: str = ""
    versions: List[AgentVersion] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)

    def add_version(self, version: AgentVersion) -> AgentVersion:
        version.agent_id = self.id
        self.versions.append(version)
        return version


# ---------------------------------------------------------------------------
# 2. EvaluationDataset & TestCase
# ---------------------------------------------------------------------------

@dataclass(init=False)
class TestCase:
    __test__ = False  # Prevent pytest from attempting to collect this class as a test suite

    test_id: str
    user_input: str
    name: str = ""
    description: str = ""
    expected_behavior: str = ""
    expected_answer: Optional[str] = None
    expected_tools: List[str] = field(default_factory=list)
    expected_tool_sequence: List[str] = field(default_factory=list)
    expected_mcp_server: Optional[str] = None
    expected_mcp_tools: List[str] = field(default_factory=list)
    expected_mcp_tool_sequence: List[str] = field(default_factory=list)
    expected_arguments: Optional[Dict[str, Any]] = None
    expected_tool_arguments: Optional[Dict[str, Any]] = None
    max_tool_calls: Optional[int] = None
    forbidden_tools: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    latency_budget: float = 5000.0
    mcp_latency_budget: float = 3000.0
    expected_output_schema: Optional[Dict[str, Any]] = None
    evaluation_metrics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"             # "easy" | "medium" | "hard"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_generate_id)

    def __init__(
        self,
        test_id: Optional[str] = None,
        user_input: Optional[str] = None,
        task_id: Optional[str] = None,
        query: Optional[str] = None,
        name: str = "",
        description: str = "",
        expected_behavior: str = "",
        expected_answer: Optional[str] = None,
        expected_tools: Optional[List[str]] = None,
        expected_tool: Optional[str] = None,
        expected_tool_sequence: Optional[List[str]] = None,
        expected_mcp_server: Optional[str] = None,
        expected_mcp_tools: Optional[List[str]] = None,
        expected_mcp_tool_sequence: Optional[List[str]] = None,
        expected_arguments: Optional[Dict[str, Any]] = None,
        expected_tool_arguments: Optional[Dict[str, Any]] = None,
        max_tool_calls: Optional[int] = None,
        forbidden_tools: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
        latency_budget: float = 5000.0,
        mcp_latency_budget: float = 3000.0,
        expected_output_schema: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        difficulty: str = "medium",
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ):
        self.test_id = str(test_id or task_id or _generate_id())
        self.user_input = str(user_input if user_input is not None else (query if query is not None else ""))
        self.name = name or f"Test {self.test_id}"
        self.description = description
        self.expected_behavior = expected_behavior
        self.expected_answer = expected_answer
        if expected_tools is not None:
            self.expected_tools = expected_tools
        elif expected_tool is not None:
            self.expected_tools = [expected_tool] if expected_tool else []
        else:
            self.expected_tools = []
        self.expected_tool_sequence = expected_tool_sequence or (list(self.expected_tools) if self.expected_tools else [])
        self.expected_mcp_server = expected_mcp_server or (metadata.get("expected_mcp_server") if metadata else None)
        self.expected_mcp_tools = expected_mcp_tools or (metadata.get("expected_mcp_tools", []) if metadata else [])
        self.expected_mcp_tool_sequence = expected_mcp_tool_sequence or (list(self.expected_mcp_tools) if self.expected_mcp_tools else [])
        self.expected_arguments = expected_arguments or (metadata.get("expected_arguments") if metadata else None)
        self.expected_tool_arguments = expected_tool_arguments or (metadata.get("expected_tool_arguments") if metadata else None) or self.expected_arguments
        self.max_tool_calls = max_tool_calls
        self.forbidden_tools = forbidden_tools or []
        self.expected_keywords = expected_keywords or []
        self.latency_budget = float(latency_budget)
        self.mcp_latency_budget = float(mcp_latency_budget)
        self.expected_output_schema = expected_output_schema
        self.tags = tags or []
        self.difficulty = difficulty
        self.enabled = bool(enabled)
        self.metadata = metadata or {}
        self.id = id or _generate_id()

        if evaluation_metrics:
            self.evaluation_metrics = evaluation_metrics
        else:
            metrics = ["latency"]
            if self.expected_tools:
                metrics.append("tool_selection")
            if self.expected_mcp_server or self.expected_mcp_tools:
                metrics.append("mcp_tool_selection")
            if self.expected_keywords:
                metrics.append("keyword_groundedness")
            if self.expected_answer:
                metrics.append("exact_answer")
            self.evaluation_metrics = metrics

    # --- Backward compatibility aliases ---
    @property
    def task_id(self) -> str:
        return self.test_id

    @task_id.setter
    def task_id(self, val: str):
        self.test_id = val

    @property
    def query(self) -> str:
        return self.user_input

    @query.setter
    def query(self, val: str):
        self.user_input = val

    @property
    def expected_tool(self) -> Optional[str]:
        return self.expected_tools[0] if self.expected_tools else None

    @expected_tool.setter
    def expected_tool(self, val: Optional[str]):
        self.expected_tools = [val] if val else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        test_id = data.get("test_id") or data.get("task_id") or _generate_id()
        user_input = data.get("user_input") or data.get("query") or ""
        expected_tools = data.get("expected_tools", [])
        if not expected_tools and data.get("expected_tool"):
            expected_tools = [data["expected_tool"]]

        return cls(
            test_id=test_id,
            user_input=user_input,
            name=data.get("name") or f"Task {test_id}",
            description=data.get("description", ""),
            expected_behavior=data.get("expected_behavior", ""),
            expected_answer=data.get("expected_answer"),
            expected_tools=expected_tools,
            expected_tool_sequence=data.get("expected_tool_sequence"),
            expected_mcp_server=data.get("expected_mcp_server"),
            expected_mcp_tools=data.get("expected_mcp_tools", []),
            expected_mcp_tool_sequence=data.get("expected_mcp_tool_sequence", []),
            expected_arguments=data.get("expected_arguments"),
            expected_tool_arguments=data.get("expected_tool_arguments"),
            max_tool_calls=data.get("max_tool_calls"),
            forbidden_tools=data.get("forbidden_tools", []),
            expected_keywords=data.get("expected_keywords", []),
            latency_budget=float(data.get("latency_budget", 5000.0)),
            mcp_latency_budget=float(data.get("mcp_latency_budget", 3000.0)),
            expected_output_schema=data.get("expected_output_schema"),
            evaluation_metrics=data.get("evaluation_metrics", []),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
            id=data.get("id", _generate_id()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "task_id": self.test_id,
            "name": self.name,
            "user_input": self.user_input,
            "query": self.user_input,
            "description": self.description,
            "expected_behavior": self.expected_behavior,
            "expected_answer": self.expected_answer,
            "expected_tools": self.expected_tools,
            "expected_tool": self.expected_tool,
            "expected_tool_sequence": self.expected_tool_sequence,
            "expected_mcp_server": self.expected_mcp_server,
            "expected_mcp_tools": self.expected_mcp_tools,
            "expected_mcp_tool_sequence": self.expected_mcp_tool_sequence,
            "expected_arguments": self.expected_arguments,
            "expected_tool_arguments": self.expected_tool_arguments,
            "max_tool_calls": self.max_tool_calls,
            "forbidden_tools": self.forbidden_tools,
            "expected_keywords": self.expected_keywords,
            "latency_budget": self.latency_budget,
            "mcp_latency_budget": self.mcp_latency_budget,
            "expected_output_schema": self.expected_output_schema,
            "evaluation_metrics": self.evaluation_metrics,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Captured Tool Call Telemetry
# ---------------------------------------------------------------------------

@dataclass
class CapturedToolCall:
    """Structured capture of an individual tool call execution telemetry.

    Captures all 10 standard dimensions + MCP context:
      - tool_name: Name of tool invoked
      - expected_tool: Expected tool for this step or task
      - arguments: Arguments passed to the tool
      - expected_arguments: Expected arguments (if specified in test case)
      - execution_status: 'success' | 'error' | 'failed' | 'timeout'
      - response: Return output of the tool call
      - latency: Tool execution latency in ms
      - retry_count: Number of consecutive/cumulative retries with identical arguments
      - error: Error message or exception details if failed
      - sequence_position: 1-indexed order within tool execution sequence
      - is_mcp: Whether this call is an MCP tool invocation
      - mcp_server: Name of the MCP server providing this tool
      - expected_mcp_server: Expected MCP server (if specified)
    """
    tool_name: str
    sequence_position: int = 1
    expected_tool: Optional[str] = None
    arguments: Any = field(default_factory=dict)
    expected_arguments: Optional[Any] = None
    execution_status: str = "success"
    response: Any = ""
    latency: float = 0.0
    retry_count: int = 0
    error: Optional[str] = None
    is_unnecessary: bool = False
    unnecessary_reason: Optional[str] = None
    is_mcp: bool = False
    mcp_server: Optional[str] = None
    expected_mcp_server: Optional[str] = None
    span_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return self.latency

    @latency_ms.setter
    def latency_ms(self, val: float):
        self.latency = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "expected_tool": self.expected_tool,
            "arguments": self.arguments,
            "expected_arguments": self.expected_arguments,
            "execution_status": self.execution_status,
            "response": self.response,
            "latency": round(self.latency, 2),
            "latency_ms": round(self.latency, 2),
            "retry_count": self.retry_count,
            "error": self.error,
            "sequence_position": self.sequence_position,
            "is_unnecessary": self.is_unnecessary,
            "unnecessary_reason": self.unnecessary_reason,
            "is_mcp": self.is_mcp,
            "mcp_server": self.mcp_server,
            "expected_mcp_server": self.expected_mcp_server,
            "span_id": self.span_id,
        }


# Alias for CapturedToolCall
ToolCallTelemetry = CapturedToolCall


@dataclass
class ToolExecutionSummary:
    """Consolidated summary of all tool calls and execution sequence across a trace."""
    tool_calls: List[CapturedToolCall] = field(default_factory=list)
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    unnecessary_tool_calls_count: int = 0
    total_tool_latency_ms: float = 0.0
    actual_tool_sequence: List[str] = field(default_factory=list)
    expected_tool_sequence: List[str] = field(default_factory=list)
    unnecessary_flags: List[str] = field(default_factory=list)
    retry_loops_detected: int = 0
    tools_called_names: List[str] = field(default_factory=list)
    is_selection_accurate: bool = True
    selection_reason: str = ""
    mcp_tool_calls_count: int = 0
    mcp_servers_involved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "unnecessary_tool_calls_count": self.unnecessary_tool_calls_count,
            "total_tool_latency_ms": round(self.total_tool_latency_ms, 2),
            "actual_tool_sequence": self.actual_tool_sequence,
            "expected_tool_sequence": self.expected_tool_sequence,
            "unnecessary_flags": self.unnecessary_flags,
            "retry_loops_detected": self.retry_loops_detected,
            "tools_called_names": self.tools_called_names,
            "is_selection_accurate": self.is_selection_accurate,
            "selection_reason": self.selection_reason,
            "mcp_tool_calls_count": self.mcp_tool_calls_count,
            "mcp_servers_involved": self.mcp_servers_involved,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class EvaluationDataset:
    name: str
    id: str = field(default_factory=_generate_id)
    description: str = ""
    version: str = "1.0"
    test_cases: List[TestCase] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _dataset_id: Optional[str] = None

    @property
    def dataset_id(self) -> str:
        return self._dataset_id or self.id

    @dataset_id.setter
    def dataset_id(self, val: str):
        self._dataset_id = val

    def add_test_case(self, case: TestCase):
        # Prevent duplicates
        if not any(c.test_id == case.test_id for c in self.test_cases):
            self.test_cases.append(case)

    def remove_test_case(self, test_id: str) -> bool:
        initial_len = len(self.test_cases)
        self.test_cases = [c for c in self.test_cases if c.test_id != test_id]
        return len(self.test_cases) < initial_len

    def filter_by_tags(self, tags: List[str]) -> List[TestCase]:
        tag_set = set(tags)
        return [tc for tc in self.test_cases if tag_set.intersection(set(tc.tags))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "test_case_ids": [tc.test_id for tc in self.test_cases],
            "total_test_cases": len(self.test_cases),
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        ds = cls(
            name=data.get("name", "Unnamed Dataset"),
            id=data.get("id") or data.get("dataset_id") or _generate_id(),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
            _dataset_id=data.get("dataset_id") or data.get("id"),
        )
        for tc_data in data.get("test_cases", []):
            if isinstance(tc_data, dict):
                ds.add_test_case(TestCase.from_dict(tc_data))
            elif isinstance(tc_data, TestCase):
                ds.add_test_case(tc_data)
        return ds


# ---------------------------------------------------------------------------
# 3. Span & Trace (Observability Telemetry)
# ---------------------------------------------------------------------------

@dataclass
class Span:
    step_type: str = "agent"               # "agent" | "llm" | "tool" | "mcp_tool" | "mcp_server" | "retrieval" | "embedding" | "memory" | "planning" | "final_answer" | "error"
    span_id: str = field(default_factory=_generate_id)
    parent_span_id: Optional[str] = None
    run_id: Optional[Any] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    latency_ms: float = 0.0
    status: str = "success"                # "success" | "error" | "running"
    tool_name: Optional[str] = None
    mcp_server: Optional[str] = None
    name: Optional[str] = None
    input_data: str = ""
    output_data: str = ""
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None
    cost_usd: float = 0.0
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)

    def __post_init__(self):
        if not self.mcp_server and self.attributes:
            self.mcp_server = self.attributes.get("mcp_server") or self.attributes.get("server_name")
        if not self.operation_name:
            if self.mcp_server and self.tool_name:
                self.operation_name = f"MCP Tool: {self.tool_name} ({self.mcp_server})"
            elif self.mcp_server:
                self.operation_name = f"MCP Server: {self.mcp_server}"
            elif self.tool_name:
                self.operation_name = f"Tool: {self.tool_name}"
            else:
                self.operation_name = self.name or f"{self.step_type.capitalize()} Operation"
        if not self.name:
            self.name = self.operation_name
        if self.end_time and self.start_time and not self.latency_ms:
            self.latency_ms = (self.end_time - self.start_time) * 1000

    @property
    def duration_ms(self) -> float:
        return self.latency_ms

    @duration_ms.setter
    def duration_ms(self, val: float):
        self.latency_ms = val

    @property
    def is_mcp(self) -> bool:
        return bool(
            self.mcp_server
            or (self.attributes and (self.attributes.get("is_mcp") or self.attributes.get("mcp_server")))
            or (self.step_type and self.step_type.lower() in ["mcp_tool", "mcp_server", "mcp_call", "mcp"])
        )

    @property
    def span_type(self) -> str:
        # Standardize step_type into supported span types
        st = (self.step_type or "agent").lower()
        if st in ["mcp_server", "mcp_host"]:
            return "mcp_server"
        if st in ["mcp_tool", "mcp_call", "mcp"]:
            return "mcp_tool"
        if self.mcp_server or (self.attributes and (self.attributes.get("is_mcp") or self.attributes.get("mcp_server"))):
            if st in ["tool_call", "tool"]:
                return "mcp_tool"
        if st in ["llm_call", "llm"]:
            return "llm"
        if st in ["tool_call", "tool"]:
            return "tool"
        if st in ["retrieval"]:
            return "retrieval"
        if st in ["agent", "root"]:
            return "agent"
        if st in ["embedding"]:
            return "embedding"
        if st in ["memory"]:
            return "memory"
        if st in ["planning"]:
            return "planning"
        if st in ["final_answer"]:
            return "final_answer"
        if st in ["error"]:
            return "error"
        return st

    @property
    def input(self) -> str:
        return self.input_data

    @property
    def output(self) -> str:
        return self.output_data

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.attributes

    def sanitize(self) -> "Span":
        """In-place secret redaction on span inputs, outputs, errors, and metadata."""
        from src.security.redactor import SecretRedactor
        return SecretRedactor.redact_span(self)


@dataclass
class Trace:
    task_id: str
    query: str
    trace_id: str = field(default_factory=_generate_id)
    spans: List[Span] = field(default_factory=list)
    final_answer: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    is_mock: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def log_span(self, span: Span):
        self.spans.append(span)

    # Backward-compatible alias for existing TracedStep callers
    def log_step(self, step: Any):
        if isinstance(step, Span):
            self.spans.append(step)
        else:
            # Convert legacy TracedStep or duck-typed object
            self.spans.append(Span(
                step_type=getattr(step, "step_type", "step"),
                tool_name=getattr(step, "tool_name", None),
                input_data=getattr(step, "input_data", ""),
                output_data=getattr(step, "output_data", ""),
                latency_ms=getattr(step, "latency_ms", 0.0),
                input_tokens=getattr(step, "input_tokens", 0),
                output_tokens=getattr(step, "output_tokens", 0),
            ))

    def finish(self, final_answer: str):
        self.final_answer = final_answer
        self.end_time = time.time()

    def sanitize(self) -> "Trace":
        """In-place secret redaction on query, final answer, metadata, and all spans."""
        from src.security.redactor import SecretRedactor
        return SecretRedactor.redact_trace(self)

    @property
    def steps(self) -> List[Span]:
        """Backward-compatibility alias for code reading trace.steps."""
        return self.spans

    @property
    def latency_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        span_sum = sum(s.latency_ms for s in self.spans)
        if span_sum > 0:
            return span_sum
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @latency_ms.setter
    def latency_ms(self, val: float):
        if self.start_time is None:
            self.start_time = time.time()
        self.end_time = self.start_time + (float(val) / 1000.0)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def tools_called(self) -> List[str]:
        return [s.tool_name for s in self.spans if (s.step_type in ["tool_call", "tool"] or s.span_type == "tool") and s.tool_name]


# ---------------------------------------------------------------------------
METRIC_DISPLAY_NAMES = {
    "task_success": "Task Success",
    "Task Success": "Task Success",
    "tool_accuracy": "Tool Accuracy",
    "Tool Accuracy": "Tool Accuracy",
    "tool_selection_accuracy": "Tool Accuracy",
    "tool_selection": "Tool Accuracy",
    "answer_correctness": "Answer Correctness",
    "Answer Correctness": "Answer Correctness",
    "exact_answer_correctness": "Answer Correctness",
    "exact_answer": "Answer Correctness",
    "semantic_answer_similarity": "Answer Correctness",
    "semantic_answer": "Answer Correctness",
    "groundedness": "Groundedness",
    "Groundedness": "Groundedness",
    "keyword_groundedness": "Groundedness",
    "context_groundedness": "Groundedness",
    "latency": "Latency",
    "Latency": "Latency",
    "latency_budget": "Latency",
}


@dataclass
class EvaluationResult:
    metric_name: str
    score: float                           # 0.0 to 1.0
    passed: bool
    details: str = ""                      # Human-readable explanation
    threshold: float = 1.0
    explanation: str = ""
    evidence: Any = field(default_factory=dict)
    evaluator_type: str = "deterministic"  # "deterministic" | "heuristic" | "llm_based" | "mock" | "semantic" | "budget"
    evaluation_type: str = "deterministic" # "deterministic" | "heuristic" | "llm_based" | "mock"
    is_mock: bool = False
    run_id: Optional[int] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.explanation and self.details:
            self.explanation = self.details
        elif not self.details and self.explanation:
            self.details = self.explanation

        # Ensure evaluation_type taxonomy synchronization
        if self.is_mock or self.evaluator_type == "mock" or self.evaluation_type == "mock":
            self.is_mock = True
            self.evaluation_type = "mock"
            self.evaluator_type = "mock"
        elif self.evaluator_type in ["semantic", "heuristic"]:
            self.evaluation_type = "heuristic"
        elif self.evaluator_type in ["llm_judge", "model_based", "llm_based"]:
            self.evaluation_type = "llm_based"
        elif not self.evaluation_type or self.evaluation_type == "deterministic":
            if self.evaluator_type not in ["deterministic", "budget", "behavioral", "structural"]:
                self.evaluation_type = self.evaluator_type
            else:
                self.evaluation_type = "deterministic"

    @property
    def metric(self) -> str:
        return METRIC_DISPLAY_NAMES.get(self.metric_name, self.metric_name)

    @metric.setter
    def metric(self, val: str):
        self.metric_name = val

    def to_dict(self) -> Dict[str, Any]:
        display_metric = METRIC_DISPLAY_NAMES.get(self.metric_name, self.metric_name)
        data = {
            "metric": display_metric,
            "metric_name": self.metric_name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "explanation": self.explanation or self.details,
            "evidence": self.evidence,
            "evaluation_type": self.evaluation_type,
            "evaluator_type": self.evaluator_type,
            "is_mock": self.is_mock,
        }
        if isinstance(self.evidence, dict):
            if "expected" in self.evidence:
                data["expected"] = self.evidence["expected"]
            if "actual" in self.evidence:
                data["actual"] = self.evidence["actual"]
        return data

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(f"EvaluationResult has no key '{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass
class EvaluationReport:
    experiment_id: str
    experiment_name: str
    agent_name: str
    dataset_name: str
    total_test_cases: int
    total_checks: int
    passed_checks: int
    overall_pass_rate: float               # Percentage 0.0 - 100.0
    metric_breakdown: Dict[str, Dict[str, Any]]
    total_latency_ms: float
    avg_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    weighted_score: float = 0.0            # Weighted composite score 0.0 - 100.0
    overall_score: float = 0.0             # Overall score 0.0 - 100.0
    failure_reasons: List[str] = field(default_factory=list)
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    is_mock: bool = False


# ---------------------------------------------------------------------------
# 5. Experiment
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    id: str = field(default_factory=_generate_id)
    description: str = ""
    agent_id: str = ""
    agent_name: str = "DemoAgent"
    agent_version: str = "1.0"
    model: str = ""
    prompt_version: str = "v1.0"
    dataset_name: str = "golden_tasks"
    dataset_version: str = "v1.0"
    environment: str = "production"        # "production" | "staging" | "ci" | "development"
    created_at: datetime = field(default_factory=_utcnow)
    status: str = "completed"              # "pending" | "running" | "completed" | "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)

