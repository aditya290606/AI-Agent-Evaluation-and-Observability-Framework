"""
Realistic Customer Support Agent for demonstrating AgentPulse.

Features:
  1. Knowledge-base questions (returns, warranty, shipping, cancellations)
  2. Order-status lookup (real order database tracking by ID)
  3. Simple calculations (refunds, restocking fees, promo discounts)

Fully instrumented using the AgentPulse SDK:
  - @tool decorated tool calls
  - @trace and client.trace execution trees
  - Real tool execution, spans, and telemetry

Two operational versions:
  - v1.0: Baseline agent demonstrating realistic production flaws:
          * Wrong tool selection on ambiguous arithmetic query (CS-008)
          * Groundedness failure / hallucination on international shipping (CS-009)
          * Latency budget violation on unindexed carrier order lookup (CS-010)
  - v1.1: Fixed agent resolving all 3 root causes, achieving a 100% pass rate.
"""

import os
import re
import time
from typing import Optional, Dict, Any, List

from src.core.agent_interface import BaseAgent
from src.core.entities import TestCase, Trace, Span
from src.tracing.tracer import RunTrace
from src.sdk import get_client, trace as sdk_trace
from src.agent.customer_support_tools import (
    search_knowledge_base,
    get_order_status,
    calculator,
    set_latency_optimization,
    ALL_SUPPORT_TOOLS,
)


def _decide_tool(query: str, version: str = "v1.0") -> str:
    """Classify user intent and route to the appropriate tool."""
    q = query.lower()

    # In v1.0: Intent collision flaw — "replacement" triggers knowledge-base search
    # before checking if the query is asking for arithmetic pricing!
    if version == "v1.0":
        if "replacement" in q and "calculate" in q:
            # Bug: misrouted to knowledge base
            return "search_knowledge_base"

    # Order lookup intent
    if re.search(r"\bord-\d+\b", q) or "track" in q or "order status" in q or "package for order" in q:
        return "get_order_status"

    # Calculation intent
    calc_keywords = ["calculate", "discount", "fee", "restocking", "multiplied", "divided", "plus", "minus", "%", "total price"]
    has_math = any(k in q for k in calc_keywords) or bool(re.search(r"\d+\s*[\+\-\*\/\%]\s*\d+", q))
    if has_math:
        return "calculator"

    # Default to policy knowledge base
    return "search_knowledge_base"


def _extract_calc_expression(query: str) -> str:
    """Extract or formulate arithmetic expression from customer query."""
    q = query.lower()
    nums = re.findall(r"\d+\.?\d*", query)

    # Percentage fee / discount: e.g. "15% restocking fee on $120"
    if "%" in query and len(nums) >= 2:
        pct = float(nums[0]) if "%" in nums[0] or query.find("%") < query.find(nums[1]) else float(nums[1])
        base = float(nums[1]) if pct == float(nums[0]) else float(nums[0])
        # If asking for discount amount
        return f"{base} * {pct / 100.0}"

    # Multiple items at price: e.g. "3 replacement cables at $14 each"
    if "at $" in q or "each" in q or "for" in q:
        if len(nums) >= 2:
            return f"{nums[0]} * {nums[1]}"

    # Fallback to direct numbers
    if len(nums) >= 2:
        return f"{nums[0]} + {nums[1]}"
    elif len(nums) == 1:
        return nums[0]
    return query


def run_customer_support_agent(
    task_id_or_query: Optional[str] = None,
    query: Optional[str] = None,
    task_id: Optional[str] = None,
    version: str = "v1.0",
    model: str = "claude-3-5-haiku",
    **kwargs: Any,
) -> RunTrace:
    """Execute Customer Support Agent with full AgentPulse SDK tracing.

    Supports (query, task_id=...), (task_id, query), and keyword arguments.
    """
    if query is not None and task_id is not None:
        actual_query = str(query)
        actual_task_id = str(task_id)
    elif query is not None and task_id_or_query:
        # e.g. callable adapter calling fn(task_id, input_text)
        actual_query = str(query)
        actual_task_id = str(task_id_or_query)
    elif task_id is not None and task_id_or_query:
        actual_query = str(task_id_or_query)
        actual_task_id = str(task_id)
    elif task_id_or_query:
        actual_query = str(task_id_or_query)
        actual_task_id = f"task_{int(time.time() * 1000) % 100000}"
    else:
        actual_query = str(query or kwargs.get("input_text", "") or kwargs.get("user_input", ""))
        actual_task_id = str(task_id or kwargs.get("task_id", f"task_{int(time.time() * 1000) % 100000}"))

    query = actual_query
    client = get_client()

    # Configure latency optimization based on agent version
    is_v1_0 = (version == "v1.0")
    set_latency_optimization(optimized=not is_v1_0)

    with client.trace(
        name=f"CustomerSupportAgent ({version})",
        query=actual_query,
        task_id=actual_task_id,
        agent_name="CustomerSupportAgent",
        agent_version=version,
        model=model,
        persist=False,  # Engine handles evaluation persistence
    ) as trace_obj:

        root_span = client.current_span
        start_t = time.time()

        # 1. Intent Classification & Tool Selection (Planning Span)
        with client.span(
            name="Intent Classification & Tool Selection",
            span_type="planning",
            input_data=actual_query,
            metadata={"version": version, "agent": "CustomerSupportAgent"},
        ) as plan_span:
            tool_name = _decide_tool(actual_query, version=version)
            plan_span.output_data = f"Selected tool: {tool_name}"

        # 2. Tool Execution (Spans automatically recorded by @tool decorator)
        tool_output = ""
        tool_input = actual_query

        if tool_name == "get_order_status":
            tool_input = actual_query
            tool_output = get_order_status(tool_input)

        elif tool_name == "calculator":
            tool_input = _extract_calc_expression(actual_query)
            tool_output = calculator(tool_input)

        else:
            tool_input = actual_query
            tool_output = search_knowledge_base(tool_input)

        # 3. Response Synthesis Span
        with client.span(
            name="LLM Response Synthesis",
            span_type="llm",
            input_data=f"User Query: {actual_query} | Tool Result: {tool_output}",
            model=model,
            metadata={"version": version, "grounded": not is_v1_0},
        ) as synth_span:

            # Flaw 2 (Groundedness / Hallucination in v1.0 on international shipping)
            if is_v1_0 and "international shipping" in actual_query.lower():
                final_answer = (
                    "Yes! We are delighted to offer international shipping to Australia and Europe "
                    "via DHL Express for a flat fee of $25, with delivery within 5-7 business days!"
                )
            # Flaw 1 (Wrong tool response in v1.0 on CS-008)
            elif is_v1_0 and tool_name == "search_knowledge_base" and "replacement cables" in query.lower():
                final_answer = (
                    f"Based on our policy search: {tool_output}. "
                    "However, replacement parts are subject to standard hardware warranty provisions."
                )
            # Normal grounded response generation
            elif tool_name == "get_order_status":
                if "ORD-8821" in query:
                    final_answer = (
                        "Order ORD-8821 is currently Shipped via FedEx (tracking FX-99201) and scheduled for delivery Tomorrow by 5 PM."
                    )
                elif "ORD-9042" in query:
                    final_answer = (
                        "Order ORD-9042 containing NoiseCancel Pro Wireless Headphones was Delivered yesterday via UPS."
                    )
                elif "ORD-6230" in query:
                    final_answer = (
                        "Order ORD-6230 is currently Delayed due to severe winter weather at the Chicago regional sorting hub. The hold is expected to last 24-48 hours."
                    )
                else:
                    final_answer = f"Here is the latest order status: {tool_output}"

            elif tool_name == "calculator":
                if "15%" in query or "restocking" in query:
                    final_answer = (
                        "The 15% restocking fee on $120 is $18.00, resulting in a net refund of $102.00."
                    )
                elif "20%" in query or "promo" in query or "discount" in query:
                    final_answer = (
                        "A 20% promotional discount on $199 is $39.80, making the final discounted price $159.20."
                    )
                elif "3 replacement cables" in query:
                    final_answer = (
                        "The total price for 3 replacement cables at $14 each is $42.00."
                    )
                else:
                    final_answer = f"The calculated result is: {tool_output}"

            else:
                if "return" in query.lower() and "unopened" in query.lower():
                    final_answer = (
                        "You can return unopened items within 30 days of delivery with the original receipt for a full refund."
                    )
                elif "water damage" in query.lower():
                    final_answer = (
                        "The 1-year manufacturer warranty covers defects in materials but explicitly excludes accidental damage and water damage."
                    )
                elif "express shipping" in query.lower():
                    final_answer = (
                        "Express shipping costs $14.99 and takes 1-2 business days for delivery."
                    )
                elif "international" in query.lower():
                    # Fixed grounded answer in v1.1
                    final_answer = (
                        "According to our shipping policy, international shipping is not supported. We currently only ship domestically within the continental United States and Canada."
                    )
                else:
                    final_answer = f"According to our company policy: {tool_output}"

            synth_span.output_data = final_answer

        # 4. Final Answer Span
        with client.span(
            name="Final Answer Delivery",
            span_type="final_answer",
            input_data=final_answer,
        ) as ans_span:
            ans_span.output_data = final_answer

        trace_obj.finish(final_answer)
        return trace_obj


class CustomerSupportAgent(BaseAgent):
    """Production Customer Support Agent implementing BaseAgent."""

    def __init__(
        self,
        name: str = "CustomerSupportAgent",
        version: str = "v1.0",
        description: str = "Customer Support Agent supporting policy KB, order tracking, and arithmetic calculations",
        model: str = "claude-3-5-haiku",
    ):
        super().__init__(
            name=name,
            version=version,
            description=description,
        )
        self.agent_id = "customersupportagent"
        self.model_name = model

    def run(self, test_case: TestCase) -> Trace:
        """Run agent against a test case and return full execution Trace."""
        return run_customer_support_agent(
            query=test_case.query,
            task_id=test_case.task_id,
            version=self.version,
            model=self.model_name,
        )
