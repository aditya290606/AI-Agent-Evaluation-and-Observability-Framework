"""
Demonstration script showing an External Agent using AgentPulse SDK.

Flow:
External Agent
 -> AgentPulse SDK (@observe)
 -> Backend (http://127.0.0.1:8000/v1/spans)
 -> Run (run_id, trace_id, agent_id, agent_version)
 -> Trace (hierarchical spans & steps)
 -> Evaluation (auto-scored metrics)
 -> Dashboard (visible in localhost:8501)
"""

import time
import os
import sys

# Ensure agentpulse is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentpulse import observe, AgentPulse

# Configure SDK to communicate with our Ingestion API
AgentPulse.init(
    backend_url="http://127.0.0.1:8000",
    api_key="eval-framework-internal-key",
    environment="production",
)


@observe(name="search_kb", span_type="tool")
def search_kb(query: str):
    """Sub-tool queried by the agent."""
    time.sleep(0.04)
    return {
        "articles": [
            {"id": "KB-104", "title": "Standard 30-day Refund Policy", "body": "Customers may request a full refund within 30 days of purchase for unused items."}
        ]
    }


@observe(name="query_order_status", span_type="tool")
def query_order_status(order_id: str):
    """Tool looking up order database."""
    time.sleep(0.03)
    return {
        "order_id": order_id,
        "customer": "Alex Mercer",
        "days_since_purchase": 12,
        "status": "delivered",
        "refundable": True,
    }


@observe(
    name="CustomerSupportAgent",
    agent_id="support_bot_sdk",
    agent_version="v1.2.0",
    span_type="agent",
)
def handle_customer_request(user_query: str):
    """External Agent function observed by AgentPulse."""
    print(f"[*] Processing user request: '{user_query}'")
    
    # Step 1: Query order status
    order_info = query_order_status("ORD-98214")
    
    # Step 2: Query Knowledge Base
    kb_info = search_kb("refund policy")
    
    # Step 3: Formulate answer
    policy = kb_info["articles"][0]["title"]
    answer = (
        f"Order {order_info['order_id']} is eligible for refund under our {policy}. "
        f"Since purchase was {order_info['days_since_purchase']} days ago (under 30 days), "
        f"your refund has been approved."
    )
    time.sleep(0.05)
    return answer


if __name__ == "__main__":
    print("[AgentPulse Demo] Running external SDK-observed agent...")
    result = handle_customer_request("Can I get a refund for order ORD-98214?")
    print(f"[AgentPulse Demo] Agent output: {result}")
    
    # Flush pending telemetry to backend
    AgentPulse.flush(timeout=5.0)
    print("[AgentPulse Demo] Execution complete and telemetry dispatched successfully!")
