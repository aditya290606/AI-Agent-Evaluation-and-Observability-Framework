"""
Tools available to the demo agent. Kept deliberately simple and
dependency-free (no external search API, no paid services) so the whole
project runs standalone -- the framework is the point, not this agent.

Swap these out for your own agent's real tools; nothing else in the
framework (tracing, storage, evaluation) needs to know what a tool does,
only its name, input, and output.
"""

import ast
import operator
import json
import os
from datetime import datetime, timezone
from langchain_core.tools import tool

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "knowledge_base.json")

with open(_KB_PATH, "r") as f:
    _KNOWLEDGE_BASE = json.load(f)

# --- safe arithmetic evaluator (no eval()) ---------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '245 * 18' or '1499 * 0.2'.
    Supports +, -, *, /, ** and parentheses. Returns the numeric result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal company knowledge base (refund policy, shipping,
    subscriptions, security, API limits, data retention, support, cancellation)
    and return the most relevant document's content."""
    query_terms = set(query.lower().split())
    best_doc, best_score = None, 0
    for doc in _KNOWLEDGE_BASE:
        doc_terms = set((doc["title"] + " " + doc["content"]).lower().split())
        score = len(query_terms & doc_terms)
        if score > best_score:
            best_score, best_doc = score, doc
    if best_doc is None:
        return "No relevant document found in the knowledge base."
    return f"[{best_doc['title']}] {best_doc['content']}"


@tool
def get_current_date(_: str = "") -> str:
    """Return today's date and day of the week. Takes no meaningful input."""
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %Y-%m-%d (UTC)")


ALL_TOOLS = [calculator, search_knowledge_base, get_current_date]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
