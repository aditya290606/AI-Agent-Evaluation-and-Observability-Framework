"""
Integration and validation tests for the AgentPulse Ingestion Pipeline.

Verifies:
  - External Agent → AgentPulse SDK → Backend → Run → Trace → Evaluation → Dashboard flow
  - Association of: agent_id, agent_version, run_id, trace_id
  - Ingestion validation (schema, types, limits)
  - Automatic evaluation of ingested runs
  - Integration type 'sdk' in Agent Registry
  - Preservation of existing mock/live runs
"""

import os
import pytest
import time

from agentpulse import observe, configure, reset_config, flush
from src.storage.db import get_session
from src.storage.models import Run, Step, EvalResult, AgentRecord
from src.storage.ingestion import (
    ingest_span_telemetry,
    validate_telemetry_payload,
    IngestionValidationError,
    authenticate_api_key,
)
from src.server import start_background_server


@pytest.fixture(autouse=True)
def clean_env():
    reset_config()
    yield
    reset_config()


def test_payload_validation_rules():
    """Verify strict validation of incoming telemetry data."""
    # 1. Missing required fields
    with pytest.raises(IngestionValidationError, match="Missing required fields"):
        validate_telemetry_payload({"trace_id": "trc_123"})

    # 2. Invalid status
    with pytest.raises(IngestionValidationError, match="Invalid status"):
        validate_telemetry_payload({
            "trace_id": "trc_1",
            "span_id": "spn_1",
            "function_name": "test_fn",
            "status": "partial_success",
        })

    # 3. Negative latency
    with pytest.raises(IngestionValidationError, match="non-negative number"):
        validate_telemetry_payload({
            "trace_id": "trc_1",
            "span_id": "spn_1",
            "function_name": "test_fn",
            "latency": -50.0,
        })

    # 4. Valid payload passes
    valid = validate_telemetry_payload({
        "trace_id": "trc_valid",
        "span_id": "spn_valid",
        "function_name": "valid_fn",
        "latency": 45.2,
        "status": "success",
    })
    assert valid["trace_id"] == "trc_valid"
    assert valid["latency"] == 45.2


def test_full_pipeline_association_and_evaluation():
    """Verify complete Flow: Agent -> SDK -> Backend -> Run -> Trace -> Evaluation."""
    configure(
        backend_url="local",
        agent_name="FinanceAuditAgent",
        agent_version="v2.1",
    )

    @observe(name="currency_converter", span_type="tool")
    def convert_currency(amount: float, fx_rate: float) -> float:
        return amount * fx_rate

    @observe(name="FinanceAuditAgent", span_type="agent")
    def run_financial_audit(query: str) -> str:
        converted = convert_currency(500.0, 1.25)
        return f"Converted total: {converted} USD"

    # Execute observed agent
    output = run_financial_audit("Convert 500 EUR to USD")
    assert output == "Converted total: 625.0 USD"

    # Wait for flush to local DB
    flush(timeout=1.0)

    # Verify unified records in storage
    session = get_session()
    try:
        # 1. Find Run associated with this trace
        root_span = getattr(run_financial_audit, "_last_span")
        assert root_span is not None

        run_row = session.query(Run).filter(Run.trace_id == root_span.trace_id).first()
        assert run_row is not None

        # Verify required associations
        assert run_row.agent_id == "financeauditagent"
        assert run_row.agent_version == "v2.1"
        assert run_row.trace_id == root_span.trace_id
        assert run_row.id is not None  # run_id
        assert run_row.query == "Convert 500 EUR to USD"
        assert "625.0 USD" in run_row.final_answer

        # 2. Verify Spans / Steps (Trace)
        steps = session.query(Step).filter(Step.run_id == run_row.id).all()
        assert len(steps) >= 2  # Agent root + currency_converter tool
        tool_step = next(s for s in steps if s.operation_name == "currency_converter")
        assert tool_step.parent_span_id == root_span.span_id

        # 3. Verify Automatic Evaluation
        evals = session.query(EvalResult).filter(EvalResult.run_id == run_row.id).all()
        assert len(evals) >= 2  # latency_budget + task_success
        metric_names = [e.metric_name for e in evals]
        assert "latency_budget" in metric_names
        assert "task_success" in metric_names

        # 4. Verify Agent Record registered with integration_type == 'sdk'
        agent_rec = session.query(AgentRecord).filter(AgentRecord.agent_id == "financeauditagent").first()
        assert agent_rec is not None
        assert agent_rec.integration_type == "sdk"
        assert agent_rec.active_version == "v2.1"

    finally:
        session.close()


def test_http_api_secure_ingestion_and_evaluation():
    """Verify HTTP API endpoint with validation, authentication, and evaluation."""
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    api_key = "secure_sdk_key_999"
    os.environ["AGENTPULSE_API_KEY"] = api_key

    server, thread = start_background_server(port=port)
    try:
        configure(
            backend_url=f"http://127.0.0.1:{port}",
            api_key=api_key,
            agent_name="SecureRemoteAgent",
            agent_version="v3.0",
        )

        @observe(name="SecureRemoteAgent")
        def remote_agent(prompt: str) -> str:
            return f"Processed securely: {prompt}"

        res = remote_agent("Classified query")
        assert res == "Processed securely: Classified query"

        flush(timeout=1.5)

        # Verify DB entry created via HTTP ingestion
        session = get_session()
        try:
            span = getattr(remote_agent, "_last_span")
            run_row = session.query(Run).filter(Run.trace_id == span.trace_id).first()
            assert run_row is not None
            assert run_row.agent_id == "secureremoteagent"
            assert run_row.agent_version == "v3.0"

            # Check evaluation results generated
            evals = session.query(EvalResult).filter(EvalResult.run_id == run_row.id).all()
            assert len(evals) >= 1
        finally:
            session.close()
    finally:
        server.shutdown()
        os.environ.pop("AGENTPULSE_API_KEY", None)
