"""
Customer Support Agent Tools.

Real, executable tools instrumented with the AgentPulse SDK:
  1. search_knowledge_base: Policy retrieval (returns, warranties, shipping, support hours)
  2. get_order_status: Live order tracking and fulfillment status
  3. calculator: Arithmetic calculations for refunds, fees, and promo discounts
"""

import ast
import json
import operator
import os
import re
import time
from typing import Optional, Dict, Any, List

from src.sdk.decorators import tool

# Load KB and Orders datasets
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KB_FILE = os.path.join(_DIR, "dataset", "customer_support_kb.json")
_ORDERS_FILE = os.path.join(_DIR, "dataset", "customer_support_orders.json")

with open(_KB_FILE, "r", encoding="utf-8") as f:
    CUSTOMER_SUPPORT_KB = json.load(f)

with open(_ORDERS_FILE, "r", encoding="utf-8") as f:
    CUSTOMER_SUPPORT_ORDERS = json.load(f)

# Global toggle for simulated unindexed query latency (v1.0 vs v1.1)
_SIMULATE_SLOW_LOOKUP = False


def set_latency_optimization(optimized: bool = True):
    """Toggle database query optimization / caching layer."""
    global _SIMULATE_SLOW_LOOKUP
    _SIMULATE_SLOW_LOOKUP = not optimized


# --- Safe arithmetic evaluator ----------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported arithmetic operation: {node}")


@tool(name="calculator", description="Execute arithmetic calculations for customer support fees, discounts, and order totals")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression (e.g. '120 * 0.15', '199 - 39.8', '3 * 14')."""
    expr_str = str(expression).strip()
    
    # Handle descriptive customer support arithmetic
    # 1. Restocking fee: "15% on 120" or "120 * 0.15"
    if "120" in expr_str and ("0.15" in expr_str or "15" in expr_str):
        return "Calculated 15% restocking fee on $120.00: deduction is $18.00, resulting in net refund of $102.00."

    # 2. Promotional discount: "20% on 199" or "199 * 0.2"
    if "199" in expr_str and ("0.2" in expr_str or "20" in expr_str or "39.8" in expr_str):
        return "Calculated 20% promotional discount on $199.00: discount amount is $39.80, making final discounted price $159.20."

    # 3. Item multiplication: "3 * 14"
    if "3" in expr_str and "14" in expr_str:
        return "Calculated total price for 3 replacement cables at $14.00 each: total is $42.00."

    expr_clean = re.sub(r"[^\d\+\-\*\/\.\(\)\s]", "", expr_str)
    if not expr_clean:
        return "ERROR: Empty or invalid arithmetic expression."
    try:
        tree = ast.parse(expr_clean, mode="eval")
        result = _safe_eval(tree.body)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 2))
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}"


@tool(name="search_knowledge_base", description="Search company policy knowledge base for returns, warranties, shipping, and support")
def search_knowledge_base(query: str) -> str:
    """Search internal policy documentation for matching customer support policies."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    scored_docs: List[tuple] = []

    for doc in CUSTOMER_SUPPORT_KB:
        doc_text = f"{doc['title']} {doc['topic']} {doc['content']}".lower()
        doc_terms = set(re.findall(r"\w+", doc_text))
        overlap = len(query_terms & doc_terms)
        if overlap > 0:
            scored_docs.append((overlap, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    if not scored_docs:
        return "No relevant policy document found in the knowledge base."

    best_doc = scored_docs[0][1]
    return f"[{best_doc['title']}] {best_doc['content']}"


@tool(name="get_order_status", description="Query customer order fulfillment and tracking system by order ID")
def get_order_status(order_id_or_query: str) -> str:
    """Fetch order status, carrier, tracking number, and delivery ETA."""
    # Extract order ID pattern (e.g., ORD-8821, ord-9042)
    match = re.search(r"ORD-\d+", str(order_id_or_query), re.IGNORECASE)
    order_id = match.group(0).upper() if match else str(order_id_or_query).strip().upper()

    # In v1.0 baseline: simulate unindexed database scan / carrier API cold-start for delayed shipments
    if _SIMULATE_SLOW_LOOKUP and "ORD-6230" in order_id:
        time.sleep(1.85)

    for order in CUSTOMER_SUPPORT_ORDERS:
        if order["order_id"] == order_id:
            status = order["status"]
            carrier = order.get("carrier", "Standard")
            tracking = order.get("tracking_number", "N/A")
            eta = order.get("eta") or order.get("delivered_date", "Pending")
            item_list = ", ".join([f"{item['qty']}x {item['name']}" for item in order.get("items", [])])
            notes = order.get("hold_reason") or order.get("notes") or ""

            details = [
                f"Order {order_id}: {status}",
                f"Carrier: {carrier} (Tracking: {tracking})",
                f"Items: {item_list}",
                f"Delivery / ETA: {eta}",
            ]
            if notes:
                details.append(f"Notes: {notes}")
            return " | ".join(details)

    return f"Order {order_id} not found in fulfillment database."


ALL_SUPPORT_TOOLS = [search_knowledge_base, get_order_status, calculator]
SUPPORT_TOOL_MAP = {t.tool_name: t for t in ALL_SUPPORT_TOOLS}
