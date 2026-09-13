"""
Unit tests for the pluggable LLM Judge architecture.
"""

import pytest
from src.core.entities import TestCase, Trace, Span
from src.evaluation.judges import (
    LLMJudge,
    JudgeVerdict,
    JudgeCriteria,
    JudgeEvidenceContext,
    MockJudge,
    OpenAIJudge,
    AnthropicJudge,
    get_llm_judge,
    parse_strict_judge_json,
)
from src.evaluation.evaluators import LLMJudgeEvaluator


def test_mock_judge_criteria_and_evidence():
    judge = MockJudge(threshold=0.60)
    ctx = JudgeEvidenceContext(
        query="What is the refund turnaround?",
        final_answer="Refunds take 7 business days to process.",
        expected_answer="Refunds take 7 business days.",
        retrieved_context=["Refund policy: All approved refunds require 7 business days."],
        tool_outputs=[{"tool_name": "search_kb", "output": "Refunds take 7 business days."}],
    )
    verdict = judge.evaluate(ctx)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.score >= 0.60
    assert verdict.mode == "mock"
    assert verdict.provider == "mock"
    assert verdict.evaluator_label == "LLM Judge — MOCK"

    # Verify structured criteria
    assert "correctness" in verdict.criteria
    assert "relevance" in verdict.criteria
    assert "groundedness" in verdict.criteria
    assert "completeness" in verdict.criteria
    assert verdict.criteria["correctness"] > 0.5
    assert verdict.criteria["groundedness"] > 0.5

    # Verify evidence citations
    assert len(verdict.evidence) > 0
    assert any("search_kb" in str(e) or "Refund policy" in str(e) for e in verdict.evidence)


def test_mock_labeling_guarantee_never_live():
    # Guarantee that MockJudge output is NEVER labeled as LIVE
    judge = MockJudge()
    ctx = JudgeEvidenceContext(query="Hello", final_answer="Hi there!")
    verdict = judge.evaluate(ctx)
    assert verdict.evaluator_label == "LLM Judge — MOCK"
    assert verdict.mode == "mock"
    assert "LIVE" not in verdict.evaluator_label


def test_mock_judge_empty_answer():
    judge = MockJudge()
    ctx = JudgeEvidenceContext(query="Test", final_answer="")
    verdict = judge.evaluate(ctx)
    assert verdict.passed is False
    assert verdict.score == 0.0
    assert "empty" in verdict.reason.lower()


def test_parse_strict_judge_json_clean():
    raw = """
    {
      "score": 0.85,
      "passed": true,
      "criteria": {
        "correctness": 0.9,
        "relevance": 0.8,
        "groundedness": 0.9,
        "completeness": 0.8
      },
      "reason": "Accurately reflects policy.",
      "evidence": ["7 business days verified in doc 1"]
    }
    """
    parsed = parse_strict_judge_json(raw)
    assert parsed["score"] == 0.85
    assert parsed["passed"] is True
    assert parsed["criteria"]["correctness"] == 0.9
    assert parsed["evidence"] == ["7 business days verified in doc 1"]


def test_parse_strict_judge_json_markdown_wrapped():
    raw = """Here is your evaluation:
    ```json
    {
      "score": 0.4,
      "passed": false,
      "criteria": {
        "correctness": 0.3,
        "relevance": 0.5,
        "groundedness": 0.2,
        "completeness": 0.6
      },
      "reason": "Hallucinated fee waiver.",
      "evidence": ["No fee waiver mentioned in tool output"]
    }
    ```
    Hope this helps!"""
    parsed = parse_strict_judge_json(raw)
    assert parsed["score"] == 0.4
    assert parsed["passed"] is False
    assert parsed["criteria"]["groundedness"] == 0.2
    assert "Hallucinated" in parsed["reason"]


def test_openai_judge_fallback_when_no_key():
    # OpenAIJudge with no API key must gracefully fall back to MockJudge
    judge = OpenAIJudge(api_key="", fallback_to_mock=True)
    ctx = JudgeEvidenceContext(
        query="Policy question",
        final_answer="The fee is 50 dollars.",
        expected_answer="The fee is 50 dollars.",
    )
    verdict = judge.evaluate(ctx)
    # Must fall back cleanly and explicitly be labeled as MOCK
    assert verdict.mode == "mock"
    assert verdict.evaluator_label == "LLM Judge — MOCK"
    assert "MOCK FALLBACK" in verdict.reason


def test_anthropic_judge_fallback_when_no_key():
    judge = AnthropicJudge(api_key="", fallback_to_mock=True)
    ctx = JudgeEvidenceContext(
        query="Policy question",
        final_answer="The fee is 50 dollars.",
        expected_answer="The fee is 50 dollars.",
    )
    verdict = judge.evaluate(ctx)
    assert verdict.mode == "mock"
    assert verdict.evaluator_label == "LLM Judge — MOCK"
    assert "MOCK FALLBACK" in verdict.reason


def test_llm_judge_evaluator_integration():
    evaluator = LLMJudgeEvaluator(threshold=0.70)
    tc = TestCase(
        test_id="T001",
        query="How many days does a refund take?",
        expected_answer="7 business days",
        expected_keywords=["7", "days"],
    )
    trace = Trace(task_id="T001", query=tc.query, final_answer="Refunds take 7 business days.", is_mock=True)
    trace.log_span(Span(
        step_type="tool",
        tool_name="search_knowledge_base",
        input_data="refund policy",
        output_data="Approved refunds take 7 business days.",
    ))
    res = evaluator(test_case=tc, trace=trace)

    assert res.metric_name == "llm_judge"
    assert res.score >= 0.70
    assert res.passed is True
    assert res.evaluator_type == "LLM Judge — MOCK"
    assert "criteria" in res.evidence
    assert "correctness" in res.evidence["criteria"]
    assert "evaluator_label" in res.evidence
    assert res.evidence["evaluator_label"] == "LLM Judge — MOCK"
