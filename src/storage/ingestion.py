"""
Unified Telemetry Ingestion & Automatic Evaluation Engine.

Bridges external SDK telemetry into the core AgentPulse framework:
  External Agent → AgentPulse SDK → Backend Ingestion → Run → Trace → Evaluation → Dashboard

Ensures all executions are strictly validated and associated with:
  - agent_id
  - agent_version
  - run_id
  - trace_id

Validation enforces:
  - Required fields: trace_id, span_id, function_name
  - Association fields: agent_id, agent_version (from top-level or metadata)
  - Max-length constraints on all ID fields
  - Safe character enforcement on agent_id
  - Non-negative numeric fields
"""

import hmac
import json
import logging
import os
import re
import time
from typing import Dict, Any, Tuple, Optional, List

from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, AgentRecord, AgentVersionRecord, TestCaseRecord
from src.security.redactor import SecretRedactor
from src.security.guardrails import ResourceGuardrails
from src.evaluation import metrics

logger = logging.getLogger("agentpulse.ingestion")

# Ensure database schema is migrated and up-to-date
init_db()


class IngestionValidationError(ValueError):
    """Raised when an incoming telemetry payload violates schema or constraints."""
    pass


# Safe agent_id pattern: alphanumeric, underscores, hyphens, dots
_SAFE_AGENT_ID_RE = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def _resolve_field(data: dict, field: str, fallback_field: str = None) -> Optional[str]:
    """Resolve a field from top-level data or nested metadata."""
    val = data.get(field)
    if val:
        return str(val).strip()
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        val = meta.get(field)
        if val:
            return str(val).strip()
        if fallback_field:
            val = meta.get(fallback_field)
            if val:
                return str(val).strip()
    return None


def validate_telemetry_payload(data: Any) -> Dict[str, Any]:
    """Strictly validate incoming telemetry payload against AgentPulse schema.

    Validates:
      - Required: trace_id, span_id, function_name
      - Association: agent_id, agent_version (from top-level or metadata)
      - Limits: max-length on all ID fields, safe characters on agent_id
      - Numerics: latency >= 0, valid timestamps
    """
    if not isinstance(data, dict):
        raise IngestionValidationError("Payload must be a JSON object.")

    # Required fields
    required_fields = ["trace_id", "span_id", "function_name"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        raise IngestionValidationError(f"Missing required fields: {', '.join(missing)}")

    # Type & format validations
    trace_id = str(data["trace_id"]).strip()
    span_id = str(data["span_id"]).strip()
    func_name = str(data["function_name"]).strip()

    if len(trace_id) > 128:
        raise IngestionValidationError("trace_id exceeds maximum length of 128 characters.")
    if len(span_id) > 128:
        raise IngestionValidationError("span_id exceeds maximum length of 128 characters.")
    if len(func_name) > 256:
        raise IngestionValidationError("function_name exceeds maximum length of 256 characters.")

    status = str(data.get("status", "success")).lower()
    if status not in ("success", "error"):
        raise IngestionValidationError(f"Invalid status '{status}'; must be 'success' or 'error'.")

    # --- Association fields: agent_id, agent_version, run_id ---
    # Resolve from top-level or metadata, with sensible defaults
    agent_id = _resolve_field(data, "agent_id", "agent_name")
    if agent_id:
        # Normalize: lowercase, replace spaces with underscores
        agent_id = agent_id.lower().replace(" ", "_")
        if len(agent_id) > 128:
            raise IngestionValidationError("agent_id exceeds maximum length of 128 characters.")
        if not _SAFE_AGENT_ID_RE.match(agent_id):
            raise IngestionValidationError(
                f"agent_id '{agent_id}' contains invalid characters. "
                "Only alphanumeric, underscores, hyphens, and dots are allowed."
            )
    else:
        # Derive from function_name as fallback
        agent_id = func_name.lower().replace(" ", "_")

    agent_version = _resolve_field(data, "agent_version")
    if agent_version:
        if len(agent_version) > 64:
            raise IngestionValidationError("agent_version exceeds maximum length of 64 characters.")
    else:
        agent_version = "v1.0"

    run_id = _resolve_field(data, "run_id")
    if run_id:
        if len(run_id) > 128:
            raise IngestionValidationError("run_id exceeds maximum length of 128 characters.")
    else:
        run_id = trace_id  # Default: run_id == trace_id

    agent_name = _resolve_field(data, "agent_name") or agent_id

    # Numeric fields
    try:
        latency = float(data.get("latency", 0.0))
        if latency < 0.0:
            raise ValueError()
    except (ValueError, TypeError):
        raise IngestionValidationError("Field 'latency' must be a non-negative number.")

    try:
        start_time = float(data.get("start_time", time.time()))
        end_time = float(data.get("end_time", start_time + (latency / 1000.0)))
    except (ValueError, TypeError):
        raise IngestionValidationError("Fields 'start_time' and 'end_time' must be valid timestamps.")

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": str(data["parent_span_id"]).strip() if data.get("parent_span_id") else None,
        "function_name": func_name,
        "input": data.get("input", ""),
        "output": data.get("output", ""),
        "latency": latency,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "exception_info": data.get("exception_info"),
        "span_type": str(data.get("span_type", "function")),
        "metadata": data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        # Validated association fields
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_version": agent_version,
        "run_id": run_id,
    }


def authenticate_api_key(provided_key: Optional[str]) -> bool:
    """Constant-time timing-attack safe API key verification."""
    expected_key = os.getenv("AGENTPULSE_API_KEY", "").strip()
    if not expected_key:
        return True  # Open if server has no key configured

    if not provided_key:
        return False

    return hmac.compare_digest(provided_key.strip(), expected_key)


def ingest_span_telemetry(raw_data: Any, auto_evaluate: bool = True) -> Dict[str, Any]:
    """Process, validate, persist, and evaluate an incoming SDK telemetry span."""
    # 1. Validation (now includes agent_id, agent_version, run_id)
    data = validate_telemetry_payload(raw_data)

    trace_id = data["trace_id"]
    span_id = data["span_id"]
    parent_span_id = data["parent_span_id"]
    func_name = data["function_name"]
    latency = data["latency"]
    status = data["status"]
    span_type = data["span_type"]
    meta = data["metadata"]

    # Use validated association fields
    agent_id = data["agent_id"]
    agent_name = data["agent_name"]
    agent_version = data["agent_version"]
    run_id = data["run_id"]
    model = meta.get("model") or "claude-3-5-haiku"

    # Redact credentials and truncate payload limits
    sanitized_input = ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(data["input"])))
    sanitized_output = ResourceGuardrails.truncate_output(SecretRedactor.redact_text(str(data["output"])))
    error_msg = json.dumps(SecretRedactor.redact_data(data["exception_info"])) if data["exception_info"] else None

    session = get_session()
    try:
        # 2. Ensure Agent is registered in Agent Registry with integration_type="sdk"
        existing_agent = session.query(AgentRecord).filter(
            (AgentRecord.agent_id == agent_id) | (AgentRecord.name == agent_name)
        ).first()
        if not existing_agent:
            agent_rec = AgentRecord(
                agent_id=agent_id,
                name=agent_name,
                description=f"Automated SDK Agent registered via AgentPulse Telemetry",
                framework="sdk",
                provider_model=model,
                integration_type="sdk",
                status="active",
                active_version=agent_version,
            )
            session.add(agent_rec)
            session.flush()

            ver_id = f"ver_{agent_id}_{agent_version.replace('.', '_')}"
            ver_rec = session.query(AgentVersionRecord).filter(AgentVersionRecord.version_id == ver_id).first()
            if not ver_rec:
                ver_rec = AgentVersionRecord(
                    version_id=ver_id,
                    agent_id=agent_id,
                    version=agent_version,
                    model=model,
                    configuration_metadata=json.dumps({"integration_type": "sdk"}),
                    status="active",
                )
                session.add(ver_rec)
                session.flush()
        else:
            agent_id = existing_agent.agent_id
            ver_id = f"ver_{existing_agent.agent_id}_{agent_version.replace('.', '_')}"
            ver_rec = session.query(AgentVersionRecord).filter(AgentVersionRecord.version_id == ver_id).first()
            if not ver_rec:
                ver_rec = AgentVersionRecord(
                    version_id=ver_id,
                    agent_id=existing_agent.agent_id,
                    version=agent_version,
                    model=model,
                    configuration_metadata=json.dumps({"integration_type": "sdk"}),
                    status="active",
                )
                session.add(ver_rec)
                session.flush()

        # 3. Find or create unified Run record
        #    Match by trace_id first, then run_id, then task_id
        run_row = session.query(Run).filter(Run.trace_id == trace_id).first()
        if not run_row and run_id != trace_id:
            run_row = session.query(Run).filter(Run.trace_id == run_id).first()
        if not run_row:
            run_row = session.query(Run).filter(Run.task_id == trace_id).first()

        tool_name = meta.get("tool_name") or (func_name if span_type in ("tool", "tool_call", "mcp_tool") else None)

        if not run_row:
            run_row = Run(
                task_id=trace_id,
                trace_id=trace_id,
                query=sanitized_input[:2000] if parent_span_id is None else "",
                final_answer=sanitized_output[:4000] if parent_span_id is None else "",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_version=agent_version,
                model=model,
                latency_ms=latency,
                tools_called=tool_name or "",
                is_mock=False,
            )
            session.add(run_row)
            session.flush()
        else:
            # Root span always updates query, final_answer, agent_id, and agent_version
            if parent_span_id is None:
                run_row.query = sanitized_input[:2000]
                run_row.final_answer = sanitized_output[:4000]
                run_row.agent_id = agent_id
                run_row.agent_name = agent_name
                run_row.agent_version = agent_version
                run_row.latency_ms = max(run_row.latency_ms or 0.0, latency)
            elif not run_row.final_answer:
                run_row.final_answer = sanitized_output[:4000]
            if tool_name:
                current_tools = [t for t in (run_row.tools_called or "").split(",") if t]
                if tool_name not in current_tools:
                    current_tools.append(tool_name)
                    run_row.tools_called = ",".join(current_tools)

        # 4. Insert Step record linked to run_id
        #    Auto-increment step_index based on existing steps for this run
        existing_step_count = session.query(Step).filter(Step.run_id == run_row.id).count()

        step_row = Step(
            run_id=run_row.id,
            step_index=existing_step_count,
            step_type=span_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=func_name,
            tool_name=tool_name,
            input_data=sanitized_input[:4000],
            output_data=sanitized_output[:4000],
            latency_ms=latency,
            status=status,
            error=error_msg,
            start_time=data["start_time"],
            end_time=data["end_time"],
            metadata_json=json.dumps(SecretRedactor.redact_data(meta)),
        )
        session.add(step_row)
        session.flush()

        # 5. Automated Evaluation: Run -> Trace -> Evaluation
        if auto_evaluate:
            _evaluate_run(session, run_row, latency, status)

        session.commit()

        return {
            "status": "success",
            "run_id": run_row.id,
            "trace_id": trace_id,
            "span_id": span_id,
            "agent_id": agent_id,
            "agent_version": agent_version,
            "run_id_sdk": run_id,
            "latency_ms": latency,
        }
    except Exception as e:
        session.rollback()
        logger.error("[AgentPulse] Ingestion persistence failed: %s", e)
        raise
    finally:
        session.close()


def _evaluate_run(session, run_row: Run, latency_ms: float, status: str):
    """Compute and persist automated evaluation metrics for an ingested run."""
    # Prevent duplicate eval results if spans are appended incrementally
    existing_evals = session.query(EvalResult).filter(EvalResult.run_id == run_row.id).all()
    existing_metric_names = {e.metric_name for e in existing_evals}

    # Check for matching TestCase by task_id or query
    matched_test = session.query(TestCaseRecord).filter(
        (TestCaseRecord.test_id == run_row.task_id) | (TestCaseRecord.user_input == run_row.query)
    ).first()

    checks: List[Tuple[str, float, bool, str, float, str, Dict[str, Any]]] = []

    # 1. Latency Budget Check
    budget = matched_test.latency_budget if matched_test else 5000.0
    passed_lat = latency_ms <= budget
    score_lat = 1.0 if passed_lat else max(0.0, 1.0 - ((latency_ms - budget) / budget))
    checks.append((
        "latency_budget",
        round(score_lat, 2),
        passed_lat,
        f"Execution latency {latency_ms:.0f}ms vs budget {budget:.0f}ms",
        budget,
        "budget",
        {"actual_ms": latency_ms, "budget_ms": budget},
    ))

    # 2. Task Success Check
    task_passed = (status == "success")
    checks.append((
        "task_success",
        1.0 if task_passed else 0.0,
        task_passed,
        "Agent completed without unhandled exceptions" if task_passed else "Agent execution terminated with error status",
        1.0,
        "deterministic",
        {"status": status},
    ))

    # 3. If matched golden task, evaluate tool selection & keyword groundedness
    if matched_test:
        expected_tools = json.loads(matched_test.expected_tools or "[]")
        if expected_tools and "tool_selection" not in existing_metric_names:
            called = [t for t in (run_row.tools_called or "").split(",") if t]
            tool_match = any(t in called for t in expected_tools)
            checks.append((
                "tool_accuracy",
                1.0 if tool_match else 0.0,
                tool_match,
                f"Expected tool {expected_tools}, called {called}",
                1.0,
                "deterministic",
                {"expected": expected_tools, "actual": called},
            ))

        expected_kws = json.loads(matched_test.expected_keywords or "[]")
        if expected_kws and "keyword_groundedness" not in existing_metric_names:
            ans_lower = (run_row.final_answer or "").lower()
            matched_k = [kw for kw in expected_kws if kw.lower() in ans_lower]
            kw_score = len(matched_k) / len(expected_kws) if expected_kws else 1.0
            checks.append((
                "keyword_groundedness",
                round(kw_score, 2),
                kw_score >= 0.7,
                f"Matched {len(matched_k)}/{len(expected_kws)} expected keywords",
                0.7,
                "deterministic",
                {"expected": expected_kws, "matched": matched_k},
            ))

    for metric_name, score, passed, details, thresh, etype, evid in checks:
        if metric_name not in existing_metric_names:
            session.add(EvalResult(
                run_id=run_row.id,
                metric_name=metric_name,
                score=score,
                passed=passed,
                threshold=thresh,
                evaluator_type=etype,
                details=details,
                evidence_json=json.dumps(evid),
            ))
