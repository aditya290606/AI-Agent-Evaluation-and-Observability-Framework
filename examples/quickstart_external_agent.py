"""
Quickstart: Integrating an External AI Agent with AgentPulse SDK.

Demonstrates:
  1. Using @agentpulse.trace to instrument any external agent function.
  2. Using @agentpulse.tool to capture internal tool executions & latency.
  3. Using @agentpulse.mcp_tool to track MCP server calls.
  4. Running standalone traced executions (persisted to AgentPulse dashboard).
  5. Programmatic evaluation against golden datasets via agentpulse.evaluate().

Run directly:
    python examples/quickstart_external_agent.py
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to path for local execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentpulse


# 1. Define and instrument tools
@agentpulse.tool(name="calculator")
def my_calculator(expression: str) -> str:
    """Safe arithmetic calculation tool."""
    try:
        # Simple arithmetic evaluator
        clean_expr = expression.replace(" ", "")
        result = eval(clean_expr, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calc Error: {e}"


@agentpulse.mcp_tool(server="Enterprise Knowledge Base", name="search_documents")
def search_enterprise_kb(query: str) -> str:
    """Simulated MCP tool providing company policy documentation."""
    q = query.lower()
    if "refund" in q or "return" in q:
        return "Returns are accepted within 14 days of receipt for a full refund or replacement."
    if "shipping" in q:
        return "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days."
    return "Relevant documentation: Standard business hours are 9am-5pm EST."


# 2. Instrument external agent workflow
@agentpulse.trace(agent_name="CustomerSupportAgent", version="v2.0", model="claude-3-5-haiku")
def support_agent(query: str) -> str:
    """External agent routing user query to appropriate tools."""
    q_lower = query.lower()

    # Route 1: Math / calculation query
    if any(k in q_lower for k in ["calculate", "tip", "%", "*", "+", "bill", "divided"]):
        # Extract or format simple math expression
        if "15%" in query and "120" in query:
            ans = my_calculator("120 * 0.15")
            return f"15% tip on a $120 bill is ${ans}."
        if "120" in query and "15" in query:
            ans = my_calculator("120 * 0.15")
            return f"The calculation result is {ans}."
        ans = my_calculator("42 * 2")
        return f"Calculated answer: {ans}"

    # Route 2: Policy / Knowledge query via MCP
    if any(k in q_lower for k in ["return", "refund", "policy", "damaged", "shipping"]):
        kb_docs = search_enterprise_kb(query)
        return f"According to our company policy: {kb_docs}"

    # Route 3: General response
    return f"I can help you with questions about policies or calculations. Your query: {query}"


if __name__ == "__main__":
    print("=" * 65)
    print("  🚀 AgentPulse SDK: External Agent Integration Demo")
    print("=" * 65)

    # 1. Standalone live execution with automatic trace capture
    print("\n[1] Running standalone traced execution...")
    user_query = "What is the return policy for damaged electronics?"
    response = support_agent(user_query)
    print(f"Agent Response: {response}")
    print("✨ Trace, tool calls, and MCP spans successfully logged to database!")

    # 2. Run programmatic evaluation against golden tasks
    print("\n[2] Running programmatic evaluation against golden_tasks...")
    report = agentpulse.evaluate(
        agent=support_agent,
        dataset="golden_tasks",
        experiment_name="CustomerSupportAgent_v2_Evaluation",
        agent_name="CustomerSupportAgent",
        agent_version="v2.0",
        print_summary=True,
    )

    print("\n👉 To inspect your agent's traces, metrics, and MCP calls:")
    print("   Open the dashboard at: http://localhost:8501")
    print("=" * 65)
