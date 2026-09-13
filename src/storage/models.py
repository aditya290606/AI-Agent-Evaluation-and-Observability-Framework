"""
Database schema for the Agent Evaluation & Observability Framework.

Three tables:
  - Run:         one row per end-to-end agent execution (one task)
  - Step:        one row per action inside a run (tool call, LLM call, final answer)
  - EvalResult:  one row per metric computed against a Run (tool accuracy,
                 groundedness, latency check, etc.)

This schema is deliberately generic -- it doesn't know anything about
LangGraph specifically, so the same tables work if you later swap the
agent implementation for CrewAI, AutoGen, raw function-calling, etc.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class AgentRecord(Base):
    """Registered AI Agent in the Agent Registry."""
    __tablename__ = "agents"

    agent_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    framework = Column(String, default="custom")            # "langgraph", "crewai", "autogen", "custom", "http"
    provider_model = Column(String, default="claude-3-5-haiku")
    integration_type = Column(String, default="mock")       # "local_python", "http_api", "mock"
    status = Column(String, default="active")               # "active", "disabled"
    active_version = Column(String, default="v1.0")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    versions = relationship("AgentVersionRecord", back_populates="agent", cascade="all, delete-orphan", lazy="joined")


class AgentVersionRecord(Base):
    """Versioned configuration for a registered Agent."""
    __tablename__ = "agent_versions"

    version_id = Column(String, primary_key=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    version = Column(String, nullable=False)                # e.g. "v1.0", "v1.1"
    model = Column(String, nullable=True)
    configuration_metadata = Column(Text, default="{}")    # JSON serialized dictionary
    status = Column(String, default="active")               # "active", "archived"
    created_at = Column(DateTime, default=utcnow)

    agent = relationship("AgentRecord", back_populates="versions")


class TestCaseRecord(Base):
    """Persisted evaluation test case."""
    __tablename__ = "test_cases"
    __test__ = False

    test_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    user_input = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    expected_behavior = Column(Text, nullable=True)
    expected_answer = Column(Text, nullable=True)
    expected_tools = Column(Text, default="[]")           # JSON serialized list
    forbidden_tools = Column(Text, default="[]")          # JSON serialized list
    expected_keywords = Column(Text, default="[]")        # JSON serialized list
    latency_budget = Column(Float, default=5000.0)
    expected_output_schema = Column(Text, nullable=True)  # JSON serialized schema
    evaluation_metrics = Column(Text, default="[]")       # JSON serialized list
    tags = Column(Text, default="[]")                     # JSON serialized list
    difficulty = Column(String, default="medium")         # "easy" | "medium" | "hard"
    enabled = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class DatasetRecord(Base):
    """Persisted Evaluation Dataset grouping multiple Test Cases."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, nullable=False, index=True)   # e.g. "golden_tasks", "ds_support_qa"
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String, nullable=False, default="1.0")    # e.g. "1.0", "1.1", "2.0"
    test_case_ids = Column(Text, default="[]")                 # JSON serialized list of test IDs
    metadata_json = Column(Text, default="{}")                 # tags, author, extra config
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Experiment(Base):
    """An evaluation campaign binding an AgentVersion, EvaluationDataset, and Run set."""
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    agent_id = Column(String, nullable=True, index=True)
    agent_name = Column(String, default="DemoAgent")
    agent_version = Column(String, default="1.0")
    model = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True, default="v1.0")
    dataset_name = Column(String, default="golden_tasks")
    dataset_version = Column(String, nullable=True, default="v1.0")
    environment = Column(String, default="production", index=True)   # "production" | "staging" | "ci" | "development"
    status = Column(String, default="completed")
    created_at = Column(DateTime, default=utcnow)

    runs = relationship("Run", back_populates="experiment")


class Run(Base):
    """One full execution of the agent against one task."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=True, index=True)
    agent_name = Column(String, nullable=True, default="DemoAgent")
    agent_id = Column(String, nullable=True, index=True)
    agent_version = Column(String, nullable=True)
    model = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True, default="v1.0")
    dataset_version = Column(String, nullable=True, default="v1.0")
    environment = Column(String, default="production", index=True)   # "production" | "staging" | "ci" | "development"
    task_id = Column(String, nullable=False, index=True)   # links to golden_tasks.json
    trace_id = Column(String, nullable=True, index=True)  # explicit trace identifier
    query = Column(Text, nullable=False)
    final_answer = Column(Text)
    expected_tool = Column(String)          # what the golden task says *should* be called
    tools_called = Column(Text)             # comma-separated list of tools actually called
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    latency_ms = Column(Float)
    est_cost_usd = Column(Float, default=0.0)
    actual_cost_usd = Column(Float, nullable=True)
    is_mock = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    experiment = relationship("Experiment", back_populates="runs")
    steps = relationship("Step", back_populates="run", cascade="all, delete-orphan")
    eval_results = relationship("EvalResult", back_populates="run", cascade="all, delete-orphan")


class Step(Base):
    """One action/span within a run: an LLM call, a tool call, or the final answer."""
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)     # order within the run
    step_type = Column(String, nullable=False)       # "llm_call" | "tool_call" | "final_answer"
    span_id = Column(String, nullable=True)
    parent_span_id = Column(String, nullable=True)
    operation_name = Column(String, nullable=True)
    tool_name = Column(String)                        # populated when step_type == "tool_call"
    input_data = Column(Text)
    output_data = Column(Text)
    latency_ms = Column(Float)
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    status = Column(String, default="success")
    error = Column(Text, nullable=True)
    model = Column(String, nullable=True)
    cost_usd = Column(Float, default=0.0)
    metadata_json = Column(Text, default="{}")
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    timestamp = Column(DateTime, default=utcnow)

    run = relationship("Run", back_populates="steps")


class EvalResult(Base):
    """One metric score computed against a completed Run."""
    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    metric_name = Column(String, nullable=False, index=True)      # "tool_accuracy" | "groundedness" | "latency_budget"
    score = Column(Float)                              # 0.0 - 1.0
    passed = Column(Boolean, index=True)
    threshold = Column(Float, nullable=True)
    evaluator_type = Column(String, nullable=True)
    details = Column(Text)                              # human-readable reasoning (e.g. judge's explanation)
    evidence_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)

    run = relationship("Run", back_populates="eval_results")


# Backward-compatible model aliases
EvaluationResultRow = EvalResult
StepRow = Step
SpanRow = Step
RunRow = Run
