"""
Unit and Integration Tests for Cost Observability and Model-Aware Pricing.
"""

import pytest
import pandas as pd
from src.cost.pricing import (
    ModelPricing,
    DEFAULT_MODEL_PRICING,
    FALLBACK_MODEL_PRICING,
    DEFAULT_TOOL_PRICING,
)
from src.cost.calculator import (
    CostCalculator,
    CostBreakdown,
    global_cost_calculator,
)
from src.cost.analytics import CostAnalyticsEngine
from src.core.entities import Trace, Span, TestCase
from src.evaluation.evaluators import CostBudgetEvaluator


class TestCostPricing:
    """Tests for ModelPricing and pricing catalog."""

    def test_default_catalog_coverage(self):
        """Verify standard models across major providers exist in catalog."""
        catalog = DEFAULT_MODEL_PRICING
        assert "gpt-4o" in catalog
        assert "gpt-4o-mini" in catalog
        assert "claude-3-5-sonnet" in catalog
        assert "claude-3-5-haiku" in catalog
        assert "gemini-1.5-pro" in catalog
        assert "gemini-1.5-flash" in catalog
        assert "deepseek-chat" in catalog
        assert "deepseek-reasoner" in catalog
        assert "llama-3.3-70b" in catalog

    def test_model_pricing_calculation(self):
        """Test calculation of input, output, and cached token pricing."""
        p = ModelPricing("test-model", "test_provider", input_price_per_mtoken=2.0, output_price_per_mtoken=10.0, cached_input_price_per_mtoken=0.5)
        # 1,000 uncached in + 1,000 cached in + 500 out
        # (1000/1M * 2.0) = 0.002
        # (1000/1M * 0.5) = 0.0005
        # (500/1M * 10.0) = 0.005
        # Total = 0.0075
        cost = p.calculate_cost(input_tokens=2000, output_tokens=500, cached_tokens=1000)
        assert round(cost, 6) == 0.0075


class TestCostCalculator:
    """Tests for centralized CostCalculator."""

    def test_calculate_llm_cost_openai(self):
        calc = CostCalculator()
        # gpt-4o: $2.50 / 1M in, $10.00 / 1M out
        # 10,000 in ($0.025) + 1,000 out ($0.010) = $0.035
        cb = calc.calculate_llm_cost("gpt-4o", input_tokens=10_000, output_tokens=1_000)
        assert cb.model == "gpt-4o"
        assert cb.provider == "openai"
        assert round(cb.total_cost, 4) == 0.0350
        assert cb.is_estimated is True
        assert cb.cost_type == "ESTIMATED"

    def test_calculate_llm_cost_anthropic_and_google(self):
        calc = CostCalculator()
        # claude-3-5-haiku: $0.80 / 1M in, $4.00 / 1M out
        cb_claude = calc.calculate_llm_cost("claude-3-5-haiku", input_tokens=10_000, output_tokens=2_000)
        assert cb_claude.provider == "anthropic"
        assert round(cb_claude.total_cost, 5) == 0.01600

        # gemini-1.5-flash: $0.075 / 1M in, $0.30 / 1M out
        cb_gemini = calc.calculate_llm_cost("gemini-1.5-flash", input_tokens=100_000, output_tokens=10_000)
        assert cb_gemini.provider == "google"
        assert round(cb_gemini.total_cost, 4) == 0.0105

    def test_tool_cost_calculation(self):
        calc = CostCalculator()
        # web_search default is $0.005 per invocation
        search_cost = calc.calculate_tool_cost("web_search", call_count=3)
        assert round(search_cost, 4) == 0.0150

        # free internal tool
        calc_cost = calc.calculate_tool_cost("calculator", call_count=5)
        assert calc_cost == 0.0

    def test_trace_cost_calculation_combined(self):
        calc = CostCalculator()
        trace = Trace(
            task_id="t1",
            query="Find news and summarize",
            spans=[
                Span(operation_name="web_search", tool_name="web_search", step_type="tool_call"),
                Span(operation_name="web_search", tool_name="web_search", step_type="tool_call"),
                Span(operation_name="LLM Call", step_type="llm_call", model="gpt-4o-mini", input_tokens=2000, output_tokens=400),
            ]
        )
        cb = calc.calculate_trace_cost(trace, quality_score=90.0)
        # gpt-4o-mini: 2000 in ($0.0003) + 400 out ($0.00024) = $0.00054
        # 2x web_search = $0.01000
        # total = $0.01054
        assert cb.tool_calls_count == 2
        assert round(cb.tool_api_cost, 4) == 0.0100
        assert round(cb.estimated_llm_cost, 5) == 0.00054
        assert round(cb.total_cost, 5) == 0.01054
        assert cb.cost_efficiency > 0.0

    def test_actual_provider_cost_preservation(self):
        """When actual provider cost is passed, it must be marked ACTUAL_PROVIDER, not estimated."""
        calc = CostCalculator()
        cb = calc.calculate_llm_cost("gpt-4o", input_tokens=1000, output_tokens=200, actual_cost=0.00421)
        assert cb.is_estimated is False
        assert cb.cost_type == "ACTUAL_PROVIDER"
        assert cb.total_cost == 0.00421
        assert "exact" in cb.disclaimer.lower()

    def test_fuzzy_matching_and_provider_inference(self):
        calc = CostCalculator()
        # Dated suffix matching
        p1 = calc.get_model_pricing("claude-3-5-sonnet-20241022")
        assert p1.model_id == "claude-3-5-sonnet"
        assert p1.provider == "anthropic"

        # Provider prefix matching
        p2 = calc.get_model_pricing("openai/gpt-4o-mini")
        assert p2.model_id == "gpt-4o-mini"
        assert p2.provider == "openai"

    def test_custom_pricing_registration(self):
        calc = CostCalculator()
        custom = ModelPricing("my-private-llm", "enterprise", input_price_per_mtoken=0.05, output_price_per_mtoken=0.10)
        calc.register_model_pricing(custom)
        
        cb = calc.calculate_llm_cost("my-private-llm", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cb.total_cost == 0.15
        assert cb.provider == "enterprise"


class TestCostAnalyticsEngine:
    """Tests for CostAnalyticsEngine aggregations and efficiency."""

    def test_compute_cost_summary(self):
        runs_df = pd.DataFrame([
            {"id": "r1", "task_id": "t1", "est_cost_usd": 0.010, "total_tokens": 2000, "status": "completed", "quality_score": 90.0},
            {"id": "r2", "task_id": "t2", "est_cost_usd": 0.020, "total_tokens": 4000, "status": "completed", "quality_score": 80.0},
            {"id": "r3", "task_id": "t3", "est_cost_usd": 0.015, "total_tokens": 3000, "status": "failed", "quality_score": 40.0},
        ])
        summary = CostAnalyticsEngine.compute_cost_summary(runs_df)
        assert summary["total_estimated_cost_usd"] == 0.045
        assert summary["cost_per_run_avg"] == 0.015
        # 2 successful tasks: 0.045 / 2 = 0.0225
        assert summary["cost_per_successful_task"] == 0.0225
        assert summary["total_tokens"] == 9000

    def test_compute_cost_by_model(self):
        runs_df = pd.DataFrame([
            {"id": "r1", "model": "gpt-4o", "est_cost_usd": 0.035, "total_tokens": 5000, "quality_score": 95.0},
            {"id": "r2", "model": "gpt-4o-mini", "est_cost_usd": 0.005, "total_tokens": 5000, "quality_score": 85.0},
        ])
        res = CostAnalyticsEngine.compute_cost_by_model(runs_df)
        assert len(res) == 2
        assert "cost_efficiency" in res.columns
        # gpt-4o-mini should have significantly higher cost efficiency
        mini_row = res[res["model"] == "gpt-4o-mini"].iloc[0]
        gpt4o_row = res[res["model"] == "gpt-4o"].iloc[0]
        assert mini_row["cost_efficiency"] > gpt4o_row["cost_efficiency"]

    def test_cost_vs_quality_extraction(self):
        runs_df = pd.DataFrame([
            {"id": "r1", "task_id": "t1", "model": "gpt-4o", "est_cost_usd": 0.02, "quality_score": 90.0, "latency_ms": 1000},
        ])
        df = CostAnalyticsEngine.compute_cost_vs_quality(runs_df)
        assert not df.empty
        assert "cost_efficiency" in df.columns
        assert df.iloc[0]["quality_score"] == 90.0


class TestCostBudgetEvaluator:
    """Tests for CostBudgetEvaluator with model-aware pricing."""

    def test_cost_budget_evaluator_pass(self):
        trace = Trace(
            task_id="t1",
            query="Simple test",
            spans=[Span(operation_name="LLM", step_type="llm", model="gpt-4o-mini", input_tokens=500, output_tokens=100)]
        )
        tc = TestCase(test_id="t1", user_input="Simple test")
        evaluator = CostBudgetEvaluator(cost_budget_usd=0.01)
        res = evaluator.evaluate(test_case=tc, execution_result="answer", trace=trace)
        assert res.passed is True
        assert res.score == 1.0
        assert "evidence" in dir(res) and res.evidence["is_estimated"] is True

    def test_cost_budget_evaluator_fail(self):
        trace = Trace(
            task_id="t2",
            query="Expensive test",
            spans=[Span(operation_name="LLM", step_type="llm", model="gpt-4", input_tokens=50_000, output_tokens=10_000)]
        )
        tc = TestCase(test_id="t2", user_input="Expensive test")
        # Limit $0.05 vs gpt-4 cost ~$2.10
        evaluator = CostBudgetEvaluator(cost_budget_usd=0.05)
        res = evaluator.evaluate(test_case=tc, execution_result="answer", trace=trace)
        assert res.passed is False
        assert res.score < 1.0
