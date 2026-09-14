"""
Database connection setup.

Swappable by design: change DATABASE_URL in .env from a sqlite:/// path
to a postgresql:// URL and nothing else in the codebase needs to change,
because all queries go through SQLAlchemy's ORM rather than raw SQL.
"""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from src.storage.models import Base

load_dotenv()

from sqlalchemy import event

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        DATABASE_URL = "sqlite:////tmp/eval_framework.db"
    else:
        DATABASE_URL = "sqlite:///./eval_framework.db"
elif DATABASE_URL.startswith("sqlite") and (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")):
    if "///./" in DATABASE_URL:
        DATABASE_URL = "sqlite:////tmp/eval_framework.db"

# check_same_thread=False and timeout=30.0 for robust multi-threaded SQLite concurrency.
connect_args = {"check_same_thread": False, "timeout": 30.0} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            cursor.execute("PRAGMA journal_mode=MEMORY;")
        else:
            cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=30000;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db():
    """Create all tables if they don't exist yet, and non-destructively
    migrate any missing columns on existing tables."""
    Base.metadata.create_all(bind=engine)

    # Automatic non-destructive schema migration for existing databases
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.connect() as conn:
        if "runs" in existing_tables:
            run_cols = [c["name"] for c in inspector.get_columns("runs")]
            if "experiment_id" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN experiment_id VARCHAR;"))
            if "agent_name" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN agent_name VARCHAR;"))
            if "agent_id" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN agent_id VARCHAR;"))
            if "agent_version" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN agent_version VARCHAR;"))
            if "model" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN model VARCHAR;"))
            if "prompt_version" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN prompt_version VARCHAR;"))
            if "dataset_version" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN dataset_version VARCHAR;"))
            if "environment" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN environment VARCHAR;"))
            if "est_cost_usd" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN est_cost_usd FLOAT DEFAULT 0.0;"))
            if "actual_cost_usd" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN actual_cost_usd FLOAT;"))
            if "trace_id" not in run_cols:
                conn.execute(text("ALTER TABLE runs ADD COLUMN trace_id VARCHAR;"))

        if "experiments" in existing_tables:
            exp_cols = [c["name"] for c in inspector.get_columns("experiments")]
            if "agent_id" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN agent_id VARCHAR;"))
            if "agent_name" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN agent_name VARCHAR;"))
            if "agent_version" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN agent_version VARCHAR;"))
            if "model" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN model VARCHAR;"))
            if "prompt_version" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN prompt_version VARCHAR;"))
            if "dataset_name" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN dataset_name VARCHAR;"))
            if "dataset_version" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN dataset_version VARCHAR;"))
            if "environment" not in exp_cols:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN environment VARCHAR;"))

        if "steps" in existing_tables:
            step_cols = [c["name"] for c in inspector.get_columns("steps")]
            if "span_id" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN span_id VARCHAR;"))
            if "parent_span_id" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN parent_span_id VARCHAR;"))
            if "operation_name" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN operation_name VARCHAR;"))
            if "start_time" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN start_time FLOAT;"))
            if "end_time" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN end_time FLOAT;"))
            if "status" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN status VARCHAR DEFAULT 'success';"))
            if "error" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN error TEXT;"))
            if "model" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN model VARCHAR;"))
            if "cost_usd" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN cost_usd FLOAT DEFAULT 0.0;"))
            if "metadata_json" not in step_cols:
                conn.execute(text("ALTER TABLE steps ADD COLUMN metadata_json TEXT DEFAULT '{}';"))

        if "eval_results" in existing_tables:
            eval_cols = [c["name"] for c in inspector.get_columns("eval_results")]
            if "threshold" not in eval_cols:
                conn.execute(text("ALTER TABLE eval_results ADD COLUMN threshold FLOAT;"))
            if "evaluator_type" not in eval_cols:
                conn.execute(text("ALTER TABLE eval_results ADD COLUMN evaluator_type VARCHAR;"))
            if "evidence_json" not in eval_cols:
                conn.execute(text("ALTER TABLE eval_results ADD COLUMN evidence_json TEXT DEFAULT '{}';"))

        conn.commit()

    # Seed baseline agents into the registry if empty
    session = SessionLocal()
    from src.storage.models import AgentRecord, AgentVersionRecord
    import json

    if session.query(AgentRecord).count() == 0:
        demo_agent = AgentRecord(
            agent_id="demo_react_agent",
            name="Demo ReAct Agent",
            description="Baseline LangGraph ReAct agent with Calculator, Knowledge Base search, and Date tools",
            framework="langgraph",
            provider_model="claude-3-5-haiku",
            integration_type="mock",
            status="active",
            active_version="v1.0",
        )
        v1 = AgentVersionRecord(
            version_id="ver_demo_v1_0",
            agent_id="demo_react_agent",
            version="v1.0",
            model="claude-3-5-haiku",
            configuration_metadata=json.dumps({"mode": "mock", "inject_bug": False}),
            status="active",
        )
        v1_buggy = AgentVersionRecord(
            version_id="ver_demo_v1_1",
            agent_id="demo_react_agent",
            version="v1.1-buggy",
            model="claude-3-5-haiku",
            configuration_metadata=json.dumps({"mode": "mock", "inject_bug": True}),
            status="active",
        )
        session.add(demo_agent)
        session.add(v1)
        session.add(v1_buggy)
        session.commit()

    # Seed baseline test cases from golden_tasks.json if empty
    from src.storage.models import TestCaseRecord
    if session.query(TestCaseRecord).count() == 0:
        golden_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset",
            "golden_tasks.json",
        )
        if os.path.exists(golden_path):
            with open(golden_path, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)

            for task in raw_tasks:
                tid = task.get("test_id") or task.get("task_id", "")
                name = task.get("name") or f"Task {tid}"
                user_input = task.get("user_input") or task.get("query", "")
                expected_behavior = task.get("expected_behavior", "")
                tool = task.get("expected_tool")
                tools = [tool] if tool else task.get("expected_tools", [])
                kws = task.get("expected_keywords") or []
                latency_budget = float(task.get("latency_budget", 5000.0))
                enabled = bool(task.get("enabled", True))

                tc = TestCaseRecord(
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
                session.add(tc)
            session.commit()

    session.close()


def get_session():
    """Return a new DB session. Caller is responsible for closing it."""
    return SessionLocal()

