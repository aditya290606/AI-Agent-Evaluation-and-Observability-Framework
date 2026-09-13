"""
End-to-End Verification: AgentPulse SDK → Backend → Run → Trace → Evaluation → Dashboard.

This script verifies the complete telemetry pipeline by:
  1. Starting the ingestion server on a random port
  2. Configuring the SDK to send telemetry to it
  3. Running an @observe-decorated agent with nested tool calls
  4. Flushing telemetry
  5. Querying the DB to verify Run, Steps, and EvalResults exist
  6. Printing a summary with the run_id for dashboard inspection

Usage:
    python examples/verify_sdk_integration.py

After running, start the dashboard and confirm visibility:
    streamlit run dashboard/app.py
"""

import os
import sys
import socket
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentpulse import observe, configure, flush, reset_config
from src.storage.db import init_db, get_session
from src.storage.models import Run, Step, EvalResult, AgentRecord
from src.server import start_background_server


def _find_free_port() -> int:
    """Find an available local port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -----------------------------------------------------------------------
# 1. Define the external agent with nested tool calls
# -----------------------------------------------------------------------

@observe(name="risk_calculator", span_type="tool", metadata={"tool_name": "risk_calculator"})
def risk_calculator(portfolio_value: float, risk_pct: float) -> str:
    """Calculate portfolio risk exposure."""
    exposure = portfolio_value * (risk_pct / 100.0)
    return f"Risk exposure: ${exposure:,.2f} ({risk_pct}% of ${portfolio_value:,.2f})"


@observe(name="market_lookup", span_type="tool", metadata={"tool_name": "market_lookup"})
def market_lookup(ticker: str) -> str:
    """Simulated market data lookup tool."""
    data = {
        "AAPL": "Apple Inc. — $189.25 (+1.2%)",
        "GOOGL": "Alphabet Inc. — $141.80 (-0.3%)",
        "MSFT": "Microsoft Corp. — $378.91 (+0.8%)",
    }
    return data.get(ticker.upper(), f"{ticker}: No data available")


@observe(
    name="PortfolioAdvisorAgent",
    span_type="agent",
    agent_id="portfolio_advisor_agent",
    agent_version="v3.0",
    metadata={"model": "claude-3-5-haiku"},
)
def portfolio_advisor(query: str) -> str:
    """External portfolio advisor agent with tool routing."""
    q_lower = query.lower()

    if any(k in q_lower for k in ["risk", "exposure", "volatility"]):
        calc_result = risk_calculator(250000.0, 15.0)
        return f"Based on your portfolio analysis: {calc_result}"

    if any(k in q_lower for k in ["price", "stock", "ticker", "market"]):
        lookup_result = market_lookup("AAPL")
        return f"Market data: {lookup_result}"

    return f"I can help with portfolio risk analysis and market data. Your query: {query}"


# -----------------------------------------------------------------------
# 2. Main verification flow
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  AgentPulse SDK Integration Verification")
    print("  Flow: Agent -> SDK -> Backend -> Run -> Trace -> Evaluation")
    print("=" * 70)

    # Initialize DB
    init_db()

    # Find a free port and start the ingestion server
    port = _find_free_port()
    print(f"\n[1/5] Starting ingestion server on port {port}...")
    server, thread = start_background_server(port=port)

    try:
        # Configure SDK to point at the local server
        reset_config()
        configure(
            backend_url=f"http://127.0.0.1:{port}",
            agent_name="PortfolioAdvisorAgent",
            agent_version="v3.0",
        )
        print("      SDK configured -> ingestion server")

        # Run the agent
        print("\n[2/5] Executing portfolio advisor agent...")
        query = "What is my portfolio risk exposure with 15% allocation?"
        response = portfolio_advisor(query)
        print(f"      Agent response: {response}")

        # Flush telemetry
        print("\n[3/5] Flushing telemetry to backend...")
        flush(timeout=3.0)
        time.sleep(0.5)  # Extra buffer for DB writes
        print("      Telemetry flushed")

        # Verify database records
        print("\n[4/5] Verifying database records...")
        session = get_session()
        try:
            # Find the root span's trace_id
            root_span = getattr(portfolio_advisor, "_last_span", None)
            if root_span is None:
                print("      ERROR: No span captured by @observe decorator")
                return False

            trace_id = root_span.trace_id
            print(f"      Trace ID: {trace_id}")

            # Check Run record
            run_row = session.query(Run).filter(Run.trace_id == trace_id).first()
            if run_row is None:
                print("      ERROR: No Run record found for trace_id")
                return False

            run_id = run_row.id
            print(f"      Run ID: {run_id}")
            print(f"      Agent ID: {run_row.agent_id}")
            print(f"      Agent Version: {run_row.agent_version}")
            print(f"      Query: {run_row.query[:80]}...")
            print(f"      Final Answer: {run_row.final_answer[:80]}...")
            print(f"      Latency: {run_row.latency_ms:.1f} ms")
            print(f"      Is Mock: {run_row.is_mock}")

            # Check Steps (trace spans)
            steps = session.query(Step).filter(Step.run_id == run_id).order_by(Step.step_index).all()
            print(f"\n      Steps/Spans: {len(steps)}")
            for s in steps:
                parent_info = f" (parent: {s.parent_span_id[:12]}...)" if s.parent_span_id else " (root)"
                print(f"        [{s.step_index}] {s.step_type}: {s.operation_name}{parent_info}")

            # Check step_index uniqueness
            step_indices = [s.step_index for s in steps]
            if len(step_indices) != len(set(step_indices)):
                print("      WARNING: Duplicate step_index values detected")
            else:
                print("      Step indices are unique and sequential")

            # Check Evaluation Results
            evals = session.query(EvalResult).filter(EvalResult.run_id == run_id).all()
            print(f"\n      Evaluation Results: {len(evals)}")
            for ev in evals:
                status_icon = "PASS" if ev.passed else "FAIL"
                print(f"        [{status_icon}] {ev.metric_name}: {ev.score:.2f} (threshold: {ev.threshold})")

            # Check Agent Registry
            agent_rec = session.query(AgentRecord).filter(
                AgentRecord.agent_id == run_row.agent_id
            ).first()
            if agent_rec:
                print(f"\n      Agent Registry: {agent_rec.name}")
                print(f"      Integration Type: {agent_rec.integration_type}")
                print(f"      Framework: {agent_rec.framework}")
            else:
                print("      WARNING: No AgentRecord found in registry")

            # Validation summary
            print("\n" + "=" * 70)
            checks = {
                "Run record created": run_row is not None,
                "agent_id set": bool(run_row.agent_id),
                "agent_version set": bool(run_row.agent_version),
                "trace_id set": bool(run_row.trace_id),
                "run_id set": run_row.id is not None,
                "Steps >= 2 (agent + tool)": len(steps) >= 2,
                "Step indices unique": len(step_indices) == len(set(step_indices)),
                "Evaluations generated": len(evals) >= 1,
                "Agent registered as SDK": agent_rec is not None and agent_rec.integration_type == "sdk",
                "is_mock = False": not run_row.is_mock,
            }

            all_passed = True
            for check_name, passed in checks.items():
                icon = "PASS" if passed else "FAIL"
                print(f"  [{icon}] {check_name}")
                if not passed:
                    all_passed = False

            print("=" * 70)
            if all_passed:
                print("\n  ALL CHECKS PASSED")
                print(f"\n  Run #{run_id} is now visible in the dashboard.")
                print("  To verify in the dashboard:")
                print("    1. streamlit run dashboard/app.py")
                print("    2. Navigate to 'Evaluation Runs' or 'Traces'")
                print(f"    3. Look for Run #{run_id} from agent 'PortfolioAdvisorAgent'")
                print(f"    4. Use 'SDK Only' execution mode filter to isolate SDK runs")
            else:
                print("\n  SOME CHECKS FAILED — see details above")

            return all_passed

        finally:
            session.close()

    finally:
        server.shutdown()
        reset_config()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
