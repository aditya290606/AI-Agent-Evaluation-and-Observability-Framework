"""
Tests for the RAG Evaluation Module.

Covers:
  - RetrievalContext extraction from traces with/without retrieval spans
  - All 6 RAG evaluators (passing & failing scenarios)
  - Graceful skip when no retrieval present
  - No invented scores when agent doesn't provide them
  - Edge cases: empty docs, single doc, no expected answer
"""

import pytest
from src.core.entities import TestCase, Trace, Span, EvaluationResult
from src.evaluation.rag_evaluators import (
    RetrievalContext,
    RetrievedDocument,
    extract_retrieval_context,
    ContextRelevanceEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    AnswerFaithfulnessEvaluator,
    AnswerGroundednessEvaluator,
    RetrievalLatencyEvaluator,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_trace(
    query: str = "What is the refund policy?",
    doc_contents: list = None,
    doc_ids: list = None,
    relevance_scores: list = None,
    retrieval_latency_ms: float = 50.0,
    final_answer: str = "Refunds are issued within 7 business days.",
) -> Trace:
    """Build a Trace with realistic retrieval spans."""
    if doc_contents is None:
        doc_contents = [
            "Refunds are issued within 7 business days of a return being received.",
            "Shipping takes 5-7 business days for standard delivery.",
        ]
    if doc_ids is None:
        doc_ids = [f"kb{i+1:03d}" for i in range(len(doc_contents))]
    if relevance_scores is None:
        relevance_scores = [0.6, 0.3]

    trace = Trace(task_id="T_RAG", query=query, final_answer=final_answer)

    # Tool span
    tool_span = Span(
        step_type="tool",
        span_id="spn_tool_1",
        parent_span_id="spn_root",
        tool_name="search_knowledge_base",
        operation_name="Tool: search_knowledge_base",
        input_data=query,
        output_data=doc_contents[0] if doc_contents else "",
        latency_ms=retrieval_latency_ms,
        start_time=100.0,
        end_time=100.0 + retrieval_latency_ms / 1000.0,
        status="success",
        attributes={
            "query": query,
            "retrieval_query": query,
            "selected_context": doc_contents[0] if doc_contents else "",
        },
    )
    trace.log_span(tool_span)

    # Retrieval parent span
    ret_span = Span(
        step_type="retrieval",
        span_id="spn_ret_1",
        parent_span_id="spn_tool_1",
        operation_name="Retrieval",
        input_data=query,
        output_data=f"Retrieved {len(doc_contents)} candidate documents",
        latency_ms=retrieval_latency_ms * 0.6,
        start_time=100.0,
        end_time=100.0 + retrieval_latency_ms * 0.6 / 1000.0,
        status="success",
        attributes={
            "matched_docs_count": len(doc_contents),
            "retrieval_query": query,
            "total_candidates": 8,
            "total_returned": len(doc_contents),
        },
    )
    trace.log_span(ret_span)

    # Individual document spans
    for rank, (content, doc_id, score) in enumerate(
        zip(doc_contents, doc_ids, relevance_scores), 1
    ):
        doc_span = Span(
            step_type="retrieval",
            span_id=f"spn_doc_{rank}",
            parent_span_id="spn_ret_1",
            operation_name=f"document {rank}: doc",
            input_data=f"query: {query}",
            output_data=content,
            latency_ms=retrieval_latency_ms * 0.25,
            start_time=100.0,
            end_time=100.0 + retrieval_latency_ms * 0.25 / 1000.0,
            status="success",
            attributes={
                "topic": f"Topic {rank}",
                "doc_id": doc_id,
                "relevance_score": score,
                "score_provenance": "heuristic",
            },
        )
        trace.log_span(doc_span)

    trace.finish(final_answer)
    return trace


def _make_no_retrieval_trace() -> Trace:
    """Trace with only calculator/LLM spans — no retrieval."""
    trace = Trace(task_id="T_CALC", query="What is 2+2?", final_answer="4")
    trace.log_span(Span(
        step_type="tool",
        span_id="spn_calc",
        tool_name="calculator",
        operation_name="Tool: calculator",
        input_data="2+2",
        output_data="4",
        latency_ms=1.0,
    ))
    trace.finish("4")
    return trace


# ---------------------------------------------------------------------------
# 1. RetrievalContext Extraction Tests
# ---------------------------------------------------------------------------

class TestRetrievalContextExtraction:
    """Tests for extract_retrieval_context()."""

    def test_extraction_from_retrieval_trace(self):
        trace = _make_retrieval_trace()
        ctx = extract_retrieval_context(trace)
        assert ctx is not None
        assert ctx.has_retrieval
        assert ctx.query == "What is the refund policy?"
        assert ctx.retrieval_query == "What is the refund policy?"
        assert len(ctx.retrieved_documents) == 2
        assert ctx.total_documents_retrieved == 2
        assert ctx.total_candidates_searched == 8

    def test_extraction_preserves_doc_ids(self):
        trace = _make_retrieval_trace(doc_ids=["kb001", "kb002"])
        ctx = extract_retrieval_context(trace)
        assert ctx.retrieved_documents[0].doc_id == "kb001"
        assert ctx.retrieved_documents[1].doc_id == "kb002"

    def test_extraction_preserves_relevance_scores(self):
        trace = _make_retrieval_trace(relevance_scores=[0.8, 0.2])
        ctx = extract_retrieval_context(trace)
        assert ctx.retrieved_documents[0].relevance_score == 0.8
        assert ctx.retrieved_documents[1].relevance_score == 0.2

    def test_extraction_returns_none_for_no_retrieval(self):
        trace = _make_no_retrieval_trace()
        ctx = extract_retrieval_context(trace)
        assert ctx is None

    def test_extraction_from_empty_trace(self):
        trace = Trace(task_id="T_EMPTY", query="hello")
        ctx = extract_retrieval_context(trace)
        assert ctx is None

    def test_provenance_label(self):
        trace = _make_retrieval_trace()
        ctx = extract_retrieval_context(trace)
        assert ctx.provenance == "heuristic"


# ---------------------------------------------------------------------------
# 2. ContextRelevanceEvaluator Tests
# ---------------------------------------------------------------------------

class TestContextRelevanceEvaluator:
    def test_relevant_docs_pass(self):
        tc = TestCase(test_id="T1", user_input="How many business days for refunds?")
        trace = _make_retrieval_trace(
            query="How many business days for refunds?",
            doc_contents=["Refunds are issued within 7 business days of a return."],
            relevance_scores=[0.7],
        )
        result = ContextRelevanceEvaluator(threshold=0.2).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score > 0.0
        assert "rag_context_relevance" == result.metric_name

    def test_irrelevant_docs_fail(self):
        tc = TestCase(test_id="T1", user_input="What is the refund policy?")
        trace = _make_retrieval_trace(
            query="What is the refund policy?",
            doc_contents=["Quantum mechanics describes the behavior of particles at atomic scale."],
            relevance_scores=[0.0],
        )
        result = ContextRelevanceEvaluator(threshold=0.5).evaluate(tc, trace=trace)
        assert not result.passed

    def test_skip_when_no_retrieval(self):
        tc = TestCase(test_id="T1", user_input="2+2")
        trace = _make_no_retrieval_trace()
        result = ContextRelevanceEvaluator().evaluate(tc, trace=trace)
        assert result.passed
        assert result.score == 1.0
        assert "skipped" in result.explanation.lower()


# ---------------------------------------------------------------------------
# 3. ContextPrecisionEvaluator Tests
# ---------------------------------------------------------------------------

class TestContextPrecisionEvaluator:
    def test_all_relevant(self):
        tc = TestCase(test_id="T1", user_input="refund policy")
        trace = _make_retrieval_trace(
            query="refund policy",
            doc_contents=["Refunds are processed quickly.", "Refund takes 7 days."],
            relevance_scores=[0.8, 0.5],
        )
        result = ContextPrecisionEvaluator(threshold=0.5, relevance_cutoff=0.3).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score == 1.0

    def test_mixed_relevance(self):
        tc = TestCase(test_id="T1", user_input="refund policy")
        trace = _make_retrieval_trace(
            query="refund policy",
            doc_contents=["Refunds are processed quickly.", "Quantum physics lecture notes."],
            relevance_scores=[0.8, 0.1],
        )
        result = ContextPrecisionEvaluator(threshold=0.8, relevance_cutoff=0.3).evaluate(tc, trace=trace)
        assert result.score == 0.5
        assert not result.passed
        assert len(result.evidence["irrelevant_docs"]) == 1

    def test_skip_when_no_retrieval(self):
        tc = TestCase(test_id="T1", user_input="2+2")
        trace = _make_no_retrieval_trace()
        result = ContextPrecisionEvaluator().evaluate(tc, trace=trace)
        assert result.passed and result.score == 1.0


# ---------------------------------------------------------------------------
# 4. ContextRecallEvaluator Tests
# ---------------------------------------------------------------------------

class TestContextRecallEvaluator:
    def test_full_recall(self):
        tc = TestCase(test_id="T1", user_input="refund policy", expected_keywords=["7", "business", "days"])
        trace = _make_retrieval_trace(
            doc_contents=["Refunds are issued within 7 business days."],
            relevance_scores=[0.8],
        )
        result = ContextRecallEvaluator(threshold=0.5).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score >= 0.5

    def test_partial_recall(self):
        tc = TestCase(test_id="T1", user_input="refund policy", expected_keywords=["30", "day", "grace"])
        trace = _make_retrieval_trace(
            doc_contents=["Refunds are issued within 7 business days."],
            relevance_scores=[0.8],
        )
        result = ContextRecallEvaluator(threshold=0.8).evaluate(tc, trace=trace)
        assert not result.passed
        assert len(result.evidence.get("missing_tokens", [])) > 0

    def test_skip_when_no_expected(self):
        tc = TestCase(test_id="T1", user_input="refund policy")
        trace = _make_retrieval_trace()
        result = ContextRecallEvaluator().evaluate(tc, trace=trace)
        assert result.passed
        assert "skipped" in result.explanation.lower()


# ---------------------------------------------------------------------------
# 5. AnswerFaithfulnessEvaluator Tests
# ---------------------------------------------------------------------------

class TestAnswerFaithfulnessEvaluator:
    def test_faithful_answer(self):
        tc = TestCase(test_id="T1", user_input="refund timeline")
        trace = _make_retrieval_trace(
            doc_contents=["Refunds are issued within 7 business days of a return."],
            relevance_scores=[0.8],
            final_answer="Refunds are issued within 7 business days.",
        )
        result = AnswerFaithfulnessEvaluator(threshold=0.3).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score > 0.3

    def test_unfaithful_answer(self):
        tc = TestCase(test_id="T1", user_input="refund timeline")
        trace = _make_retrieval_trace(
            doc_contents=["Shipping takes 5-7 business days."],
            relevance_scores=[0.5],
            final_answer="Refunds are processed instantly with cryptocurrency.",
        )
        result = AnswerFaithfulnessEvaluator(threshold=0.5).evaluate(tc, trace=trace)
        # Many answer tokens won't be in shipping context
        assert result.score < 0.5

    def test_skip_no_retrieval(self):
        tc = TestCase(test_id="T1", user_input="2+2")
        trace = _make_no_retrieval_trace()
        result = AnswerFaithfulnessEvaluator().evaluate(tc, trace=trace)
        assert result.passed and result.score == 1.0


# ---------------------------------------------------------------------------
# 6. AnswerGroundednessEvaluator Tests
# ---------------------------------------------------------------------------

class TestAnswerGroundednessEvaluator:
    def test_fully_grounded(self):
        tc = TestCase(test_id="T1", user_input="refund", expected_keywords=["7", "business"])
        trace = _make_retrieval_trace(
            doc_contents=["Refunds are issued within 7 business days."],
            relevance_scores=[0.8],
            final_answer="Refunds take 7 business days.",
        )
        result = AnswerGroundednessEvaluator(threshold=0.5).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score >= 0.5

    def test_hallucinated_correct_answer(self):
        """Answer contains expected keywords but context doesn't — should penalize."""
        tc = TestCase(test_id="T1", user_input="grace period", expected_keywords=["30", "day", "grace"])
        trace = _make_retrieval_trace(
            doc_contents=["Shipping takes 5-7 business days."],
            relevance_scores=[0.3],
            final_answer="The grace period is 30 days.",
        )
        result = AnswerGroundednessEvaluator(threshold=0.8).evaluate(tc, trace=trace)
        # "30" and "grace" are in answer but NOT in retrieved context about shipping
        assert len(result.evidence.get("ungrounded_in_context", [])) > 0

    def test_skip_no_keywords(self):
        tc = TestCase(test_id="T1", user_input="hello")
        trace = _make_retrieval_trace()
        result = AnswerGroundednessEvaluator().evaluate(tc, trace=trace)
        assert result.passed
        assert "skipped" in result.explanation.lower()


# ---------------------------------------------------------------------------
# 7. RetrievalLatencyEvaluator Tests
# ---------------------------------------------------------------------------

class TestRetrievalLatencyEvaluator:
    def test_within_budget(self):
        tc = TestCase(test_id="T1", user_input="refund")
        trace = _make_retrieval_trace(retrieval_latency_ms=50.0)
        result = RetrievalLatencyEvaluator(budget_ms=2000.0).evaluate(tc, trace=trace)
        assert result.passed
        assert result.score == 1.0

    def test_exceeds_budget(self):
        tc = TestCase(test_id="T1", user_input="refund")
        trace = _make_retrieval_trace(retrieval_latency_ms=5000.0)
        result = RetrievalLatencyEvaluator(budget_ms=100.0).evaluate(tc, trace=trace)
        assert not result.passed
        assert result.score < 1.0

    def test_skip_no_retrieval(self):
        tc = TestCase(test_id="T1", user_input="2+2")
        trace = _make_no_retrieval_trace()
        result = RetrievalLatencyEvaluator().evaluate(tc, trace=trace)
        assert result.passed and result.score == 1.0


# ---------------------------------------------------------------------------
# 8. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_doc_retrieval(self):
        tc = TestCase(test_id="T1", user_input="refund", expected_keywords=["7"])
        trace = _make_retrieval_trace(
            doc_contents=["Refunds take 7 business days."],
            doc_ids=["kb001"],
            relevance_scores=[0.9],
            final_answer="It takes 7 business days.",
        )
        ctx = extract_retrieval_context(trace)
        assert ctx is not None
        assert len(ctx.retrieved_documents) == 1

    def test_no_relevance_scores_provided(self):
        """When agent doesn't provide scores, evaluator uses fallback heuristic."""
        trace = Trace(task_id="T1", query="refund policy", final_answer="7 days")
        # Tool span
        trace.log_span(Span(
            step_type="tool", span_id="t1", tool_name="search_knowledge_base",
            input_data="refund policy", output_data="Refunds in 7 days",
            attributes={"retrieval_query": "refund policy"},
        ))
        # Retrieval parent
        trace.log_span(Span(
            step_type="retrieval", span_id="r1", parent_span_id="t1",
            operation_name="Retrieval", latency_ms=10.0,
            attributes={"matched_docs_count": 1},
        ))
        # Document span WITHOUT relevance_score
        trace.log_span(Span(
            step_type="retrieval", span_id="d1", parent_span_id="r1",
            operation_name="document 1: refund",
            output_data="Refunds take 7 business days.",
            attributes={"topic": "Refund"},
        ))
        trace.finish("7 days")

        ctx = extract_retrieval_context(trace)
        assert ctx is not None
        assert ctx.retrieved_documents[0].relevance_score is None  # Not invented

        # Precision evaluator should fall back to token overlap, not crash
        tc = TestCase(test_id="T1", user_input="refund policy")
        result = ContextPrecisionEvaluator(threshold=0.3, relevance_cutoff=0.1).evaluate(tc, trace=trace)
        assert result.metric_name == "rag_context_precision"
