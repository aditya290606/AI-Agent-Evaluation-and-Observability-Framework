"""
RAG (Retrieval-Augmented Generation) Evaluation Module.

Provides:
  - RetrievalContext: structured extraction of retrieval telemetry from traces
  - extract_retrieval_context(): scans trace spans for retrieval data
  - 6 RAG-specific evaluators implementing BaseEvaluator:
      1. ContextRelevanceEvaluator
      2. ContextPrecisionEvaluator
      3. ContextRecallEvaluator
      4. AnswerFaithfulnessEvaluator
      5. AnswerGroundednessEvaluator
      6. RetrievalLatencyEvaluator

All evaluators are deterministic/heuristic (token overlap). They never invent
embedding similarity or scores the underlying agent did not produce.
When no retrieval spans exist in a trace, evaluators return score=1.0 (skipped).
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict

from src.core.evaluator_interface import BaseEvaluator
from src.core.entities import TestCase, Trace, EvaluationResult


# ---------------------------------------------------------------------------
# Retrieval Context Data Model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDocument:
    """One document returned by a retrieval operation."""
    rank: int = 0
    doc_id: str = ""
    title: str = ""
    content: str = ""
    relevance_score: Optional[float] = None   # None if agent didn't provide one
    provenance: str = "unknown"                # "heuristic" | "embedding" | "bm25" | "unknown"


@dataclass
class RetrievalContext:
    """Structured retrieval telemetry extracted from trace spans."""
    query: str = ""                            # The original user query
    retrieval_query: str = ""                  # The query sent to the retrieval tool
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    selected_context: str = ""                 # Final context string passed to LLM
    retrieval_latency_ms: float = 0.0          # Time spent in retrieval spans
    total_documents_retrieved: int = 0
    total_candidates_searched: int = 0
    provenance: str = "unknown"                # Score provenance label

    @property
    def has_retrieval(self) -> bool:
        return len(self.retrieved_documents) > 0 or bool(self.retrieval_query)

    @property
    def all_context_text(self) -> str:
        """Union of all retrieved document content."""
        parts = [d.content for d in self.retrieved_documents if d.content]
        if self.selected_context and self.selected_context not in " ".join(parts):
            parts.append(self.selected_context)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Extraction Helper
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _token_set(text: str) -> set:
    return set(_normalize(text).split())


def extract_retrieval_context(trace: Trace) -> Optional[RetrievalContext]:
    """
    Scan a trace for retrieval-related spans and extract structured RAG context.

    Looks for:
      - Spans with step_type 'retrieval' or tool_name 'search_knowledge_base'
      - Document sub-spans under retrieval parent spans
      - Attributes: doc_id, topic, relevance_score, retrieval_query,
        selected_context, total_candidates, total_returned

    Returns None if no retrieval activity is found.
    """
    if not trace or not trace.spans:
        return None

    retrieval_tool_spans = []
    retrieval_parent_spans = []
    document_spans = []

    for s in trace.spans:
        span_type = getattr(s, "span_type", s.step_type)

        # Identify the tool span that performed retrieval
        if s.tool_name and "knowledge_base" in (s.tool_name or "").lower():
            retrieval_tool_spans.append(s)
        # Identify retrieval parent spans (step_type == 'retrieval' with children)
        elif span_type == "retrieval" and s.operation_name and "retrieval" in s.operation_name.lower() and "document" not in s.operation_name.lower():
            retrieval_parent_spans.append(s)
        # Identify individual document spans
        elif span_type == "retrieval" and s.operation_name and "document" in s.operation_name.lower():
            document_spans.append(s)

    # If nothing retrieval-related found, return None
    if not retrieval_tool_spans and not retrieval_parent_spans and not document_spans:
        return None

    ctx = RetrievalContext()
    ctx.query = trace.query or ""

    # Extract from the tool span
    if retrieval_tool_spans:
        tool_span = retrieval_tool_spans[0]
        ctx.retrieval_query = tool_span.input_data or ctx.query
        ctx.selected_context = tool_span.output_data or ""
        attrs = tool_span.attributes or {}
        if "selected_context" in attrs:
            ctx.selected_context = str(attrs["selected_context"])
        if "retrieval_query" in attrs:
            ctx.retrieval_query = str(attrs["retrieval_query"])

    # Extract from retrieval parent span
    if retrieval_parent_spans:
        parent = retrieval_parent_spans[0]
        attrs = parent.attributes or {}
        ctx.retrieval_latency_ms = parent.latency_ms or 0.0
        ctx.total_candidates_searched = int(attrs.get("total_candidates", 0))
        ctx.total_documents_retrieved = int(attrs.get("total_returned", attrs.get("matched_docs_count", 0)))
        if attrs.get("retrieval_query"):
            ctx.retrieval_query = str(attrs["retrieval_query"])

    # Extract individual documents
    for rank_idx, ds in enumerate(document_spans, 1):
        attrs = ds.attributes or {}
        raw_score = attrs.get("relevance_score")
        relevance_score = float(raw_score) if raw_score is not None else None
        provenance = str(attrs.get("score_provenance", "unknown"))

        doc = RetrievedDocument(
            rank=rank_idx,
            doc_id=str(attrs.get("doc_id", "")),
            title=str(attrs.get("topic", "")),
            content=ds.output_data or "",
            relevance_score=relevance_score,
            provenance=provenance,
        )
        ctx.retrieved_documents.append(doc)

    if ctx.retrieved_documents:
        ctx.total_documents_retrieved = max(ctx.total_documents_retrieved, len(ctx.retrieved_documents))
        # Determine overall provenance from documents
        provenances = {d.provenance for d in ctx.retrieved_documents if d.provenance != "unknown"}
        ctx.provenance = provenances.pop() if len(provenances) == 1 else ("mixed" if provenances else "unknown")

    # If no retrieval latency from parent span, sum from document spans
    if ctx.retrieval_latency_ms == 0.0 and document_spans:
        ctx.retrieval_latency_ms = max(ds.latency_ms or 0.0 for ds in document_spans)

    # Fallback: if no retrieval_query, use the tool span input or the trace query
    if not ctx.retrieval_query:
        ctx.retrieval_query = ctx.query

    return ctx


# ===========================================================================
# 1. Context Relevance Evaluator
# ===========================================================================

class ContextRelevanceEvaluator(BaseEvaluator):
    """
    Measures whether the retrieved documents are relevant to the user query.
    Score = average token overlap between query and each document.
    """

    def __init__(self, threshold: float = 0.3, weight: float = 1.0):
        super().__init__(
            name="rag_context_relevance",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_deterministic",
            description="Token overlap between user query and retrieved documents",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval:
            return self._skip("No retrieval detected in trace; skipped.")

        query_tokens = _token_set(ctx.retrieval_query or ctx.query)
        if not query_tokens:
            return self._skip("Empty retrieval query; skipped.")

        doc_scores = []
        for doc in ctx.retrieved_documents:
            doc_tokens = _token_set(doc.content)
            if not doc_tokens:
                doc_scores.append(0.0)
                continue
            overlap = len(query_tokens & doc_tokens)
            doc_scores.append(overlap / len(query_tokens))

        score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        score = min(score, 1.0)
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Average query-document token overlap: {score:.1%} across {len(ctx.retrieved_documents)} docs.",
            evidence={
                "query": ctx.retrieval_query,
                "doc_count": len(ctx.retrieved_documents),
                "per_doc_scores": [round(s, 3) for s in doc_scores],
                "provenance": ctx.provenance,
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 2. Context Precision Evaluator
# ===========================================================================

class ContextPrecisionEvaluator(BaseEvaluator):
    """
    Of the retrieved documents, what fraction is actually relevant?
    Uses agent-provided relevance_score if available; otherwise falls back to
    token-overlap heuristic. Does NOT invent scores.
    """

    def __init__(self, threshold: float = 0.5, relevance_cutoff: float = 0.3, weight: float = 1.0):
        super().__init__(
            name="rag_context_precision",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_deterministic",
            description="Fraction of retrieved documents deemed relevant",
        )
        self.relevance_cutoff = relevance_cutoff

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval or not ctx.retrieved_documents:
            return self._skip("No retrieval detected in trace; skipped.")

        # Determine relevance per document
        relevant_docs = []
        irrelevant_docs = []
        query_tokens = _token_set(ctx.retrieval_query or ctx.query)

        for doc in ctx.retrieved_documents:
            # Prefer agent-provided score; fallback to token overlap
            if doc.relevance_score is not None:
                is_relevant = doc.relevance_score >= self.relevance_cutoff
                used_score = doc.relevance_score
            else:
                doc_tokens = _token_set(doc.content)
                used_score = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0
                is_relevant = used_score >= self.relevance_cutoff

            if is_relevant:
                relevant_docs.append({"doc_id": doc.doc_id, "title": doc.title, "score": round(used_score, 3)})
            else:
                irrelevant_docs.append({"doc_id": doc.doc_id, "title": doc.title, "score": round(used_score, 3)})

        precision = len(relevant_docs) / len(ctx.retrieved_documents)
        passed = precision >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(precision, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"{len(relevant_docs)}/{len(ctx.retrieved_documents)} retrieved docs are relevant (cutoff={self.relevance_cutoff}).",
            evidence={
                "relevant_docs": relevant_docs,
                "irrelevant_docs": irrelevant_docs,
                "relevance_cutoff": self.relevance_cutoff,
                "provenance": ctx.provenance,
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 3. Context Recall Evaluator
# ===========================================================================

class ContextRecallEvaluator(BaseEvaluator):
    """
    Does the retrieved context cover the key terms of the expected answer?
    Score = fraction of expected-answer tokens found in the union of all
    retrieved document content.
    """

    def __init__(self, threshold: float = 0.5, weight: float = 1.0):
        super().__init__(
            name="rag_context_recall",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_deterministic",
            description="Coverage of expected answer key terms in retrieved context",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval:
            return self._skip("No retrieval detected in trace; skipped.")

        # Need expected_answer or expected_keywords to measure recall
        expected_text = ""
        if test_case and test_case.expected_answer:
            expected_text = test_case.expected_answer
        elif test_case and test_case.expected_keywords:
            expected_text = " ".join(test_case.expected_keywords)
        else:
            return self._skip("No expected answer or keywords specified; skipped.")

        expected_tokens = _token_set(expected_text)
        if not expected_tokens:
            return self._skip("Empty expected answer tokens; skipped.")

        context_tokens = _token_set(ctx.all_context_text)
        covered = expected_tokens & context_tokens
        recall = len(covered) / len(expected_tokens)
        passed = recall >= self.threshold

        missing = expected_tokens - context_tokens

        return EvaluationResult(
            metric_name=self.name,
            score=round(recall, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Context covers {len(covered)}/{len(expected_tokens)} expected answer tokens ({recall:.1%}).",
            evidence={
                "covered_tokens": sorted(covered),
                "missing_tokens": sorted(missing),
                "expected_tokens_count": len(expected_tokens),
                "context_tokens_count": len(context_tokens),
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 4. Answer Faithfulness Evaluator
# ===========================================================================

class AnswerFaithfulnessEvaluator(BaseEvaluator):
    """
    Are claims in the final answer traceable to the retrieved context?
    Score = fraction of answer tokens that appear in the retrieved context.
    This measures whether the answer stays faithful to evidence, not whether
    it matches the expected answer.
    """

    def __init__(self, threshold: float = 0.4, weight: float = 1.0):
        super().__init__(
            name="rag_answer_faithfulness",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_deterministic",
            description="Fraction of answer tokens traceable to retrieved context",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval:
            return self._skip("No retrieval detected in trace; skipped.")

        answer = str(execution_result or (trace.final_answer if trace else ""))
        if not answer.strip():
            return EvaluationResult(
                metric_name=self.name, score=0.0, passed=False,
                threshold=self.threshold, explanation="Empty final answer.",
                evaluator_type=self.evaluator_type,
            )

        answer_tokens = _token_set(answer)
        # Remove common stopwords and filler tokens from scoring
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "to", "of", "in", "for", "on", "with", "at", "by", "from",
                      "as", "into", "about", "that", "this", "it", "its", "and",
                      "or", "but", "not", "no", "if", "then", "so", "based"}
        answer_tokens = answer_tokens - stopwords
        if not answer_tokens:
            return self._skip("Answer contains only stopwords; skipped.")

        context_tokens = _token_set(ctx.all_context_text)
        grounded = answer_tokens & context_tokens
        faithfulness = len(grounded) / len(answer_tokens) if answer_tokens else 0.0
        passed = faithfulness >= self.threshold

        ungrounded = answer_tokens - context_tokens

        return EvaluationResult(
            metric_name=self.name,
            score=round(faithfulness, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"{len(grounded)}/{len(answer_tokens)} answer content tokens are grounded in retrieved context ({faithfulness:.1%}).",
            evidence={
                "grounded_tokens": sorted(grounded)[:20],
                "ungrounded_tokens": sorted(ungrounded)[:20],
                "answer_token_count": len(answer_tokens),
                "context_token_count": len(context_tokens),
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 5. Answer Groundedness Evaluator
# ===========================================================================

class AnswerGroundednessEvaluator(BaseEvaluator):
    """
    Is the answer grounded in the retrieved evidence?
    Checks that expected keywords from the test case appear in both
    the retrieved context AND the final answer. Penalizes answers
    that include expected keywords not present in retrieved docs
    (i.e., hallucinated the right answer without grounding).
    """

    def __init__(self, threshold: float = 0.5, weight: float = 1.0):
        super().__init__(
            name="rag_answer_groundedness",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_deterministic",
            description="Expected keywords grounded in both context and answer",
        )

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval:
            return self._skip("No retrieval detected in trace; skipped.")

        # Need expected keywords or expected answer
        keywords = []
        if test_case and test_case.expected_keywords:
            keywords = test_case.expected_keywords
        elif test_case and test_case.expected_answer:
            keywords = _normalize(test_case.expected_answer).split()[:10]
        else:
            return self._skip("No expected keywords or answer to verify groundedness; skipped.")

        if not keywords:
            return self._skip("Empty expected keywords; skipped.")

        answer = str(execution_result or (trace.final_answer if trace else "")).lower()
        context = ctx.all_context_text.lower()

        grounded = []       # keyword in both answer AND context
        ungrounded = []     # keyword in answer but NOT in context (hallucinated correct answer)
        missing = []        # keyword not in answer at all

        for kw in keywords:
            kw_lower = kw.lower()
            in_answer = kw_lower in answer
            in_context = kw_lower in context

            if in_answer and in_context:
                grounded.append(kw)
            elif in_answer and not in_context:
                ungrounded.append(kw)
            else:
                missing.append(kw)

        score = len(grounded) / len(keywords) if keywords else 0.0
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"{len(grounded)}/{len(keywords)} keywords grounded in both context and answer.",
            evidence={
                "grounded_keywords": grounded,
                "ungrounded_in_context": ungrounded,
                "missing_from_answer": missing,
                "total_keywords": len(keywords),
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ===========================================================================
# 6. Retrieval Latency Evaluator
# ===========================================================================

class RetrievalLatencyEvaluator(BaseEvaluator):
    """
    Evaluates whether retrieval completed within acceptable latency.
    Uses the retrieval_latency_ms from the trace, not the overall run latency.
    """

    def __init__(self, budget_ms: float = 2000.0, threshold: float = 1.0, weight: float = 1.0):
        super().__init__(
            name="rag_retrieval_latency",
            threshold=threshold,
            weight=weight,
            evaluator_type="rag_budget",
            description="Retrieval latency within configured budget",
        )
        self.budget_ms = budget_ms

    def evaluate(
        self,
        test_case: TestCase,
        execution_result: Optional[Any] = None,
        trace: Optional[Trace] = None,
    ) -> EvaluationResult:
        ctx = extract_retrieval_context(trace) if trace else None
        if not ctx or not ctx.has_retrieval:
            return self._skip("No retrieval detected in trace; skipped.")

        latency = ctx.retrieval_latency_ms
        if latency <= 0:
            return self._skip("Retrieval latency not measured; skipped.")

        score = min(1.0, self.budget_ms / latency) if latency > 0 else 1.0
        passed = latency <= self.budget_ms

        return EvaluationResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=passed,
            threshold=self.threshold,
            explanation=f"Retrieval latency: {latency:.1f}ms (budget: {self.budget_ms:.0f}ms).",
            evidence={
                "retrieval_latency_ms": round(latency, 2),
                "budget_ms": self.budget_ms,
                "within_budget": passed,
            },
            evaluator_type=self.evaluator_type,
        )

    def _skip(self, reason: str) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name, score=1.0, passed=True,
            threshold=self.threshold, explanation=reason,
            evaluator_type=self.evaluator_type,
        )


# ---------------------------------------------------------------------------
# Convenience: All RAG evaluators
# ---------------------------------------------------------------------------

ALL_RAG_EVALUATORS = [
    ContextRelevanceEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    AnswerFaithfulnessEvaluator,
    AnswerGroundednessEvaluator,
    RetrievalLatencyEvaluator,
]
