"""
Centralized Cost Calculator for Agent Evaluation & Observability.

Provides model-aware and provider-aware cost calculation, tool/API execution cost tracking,
clear distinction between Estimated and Actual Provider Costs, and cost efficiency metrics.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union

from src.cost.pricing import (
    ModelPricing,
    DEFAULT_MODEL_PRICING,
    FALLBACK_MODEL_PRICING,
    DEFAULT_TOOL_PRICING,
)


@dataclass
class CostBreakdown:
    """Detailed telemetry and breakdown of an agent execution cost."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    model: str = "unknown"
    provider: str = "unknown"
    input_token_price_per_mtoken: float = 0.0
    output_token_price_per_mtoken: float = 0.0
    estimated_llm_cost: float = 0.0
    tool_api_cost: float = 0.0
    total_cost: float = 0.0
    actual_cost: Optional[float] = None
    is_estimated: bool = True
    cost_type: str = "ESTIMATED"           # "ESTIMATED" | "ACTUAL_PROVIDER"
    cost_efficiency: float = 0.0           # quality_score / total_cost
    tool_calls_count: int = 0
    tool_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    disclaimer: str = "Estimated based on token usage and published model pricing."

    def __post_init__(self):
        if self.actual_cost is not None and self.actual_cost >= 0.0:
            self.total_cost = round(self.actual_cost, 6)
            self.is_estimated = False
            self.cost_type = "ACTUAL_PROVIDER"
            self.disclaimer = "Exact cost reported directly by the provider API."
        else:
            self.total_cost = round(self.estimated_llm_cost + self.tool_api_cost, 6)
            self.is_estimated = True
            self.cost_type = "ESTIMATED"
            self.disclaimer = "Estimated based on token usage and published model pricing. Never exact."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "model": self.model,
            "provider": self.provider,
            "input_token_price_per_mtoken": self.input_token_price_per_mtoken,
            "output_token_price_per_mtoken": self.output_token_price_per_mtoken,
            "estimated_llm_cost": round(self.estimated_llm_cost, 6),
            "tool_api_cost": round(self.tool_api_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "actual_cost": round(self.actual_cost, 6) if self.actual_cost is not None else None,
            "is_estimated": self.is_estimated,
            "cost_type": self.cost_type,
            "cost_efficiency": round(self.cost_efficiency, 2),
            "tool_calls_count": self.tool_calls_count,
            "tool_breakdown": self.tool_breakdown,
            "disclaimer": self.disclaimer,
        }


class CostCalculator:
    """Centralized calculator for model-aware LLM and tool API costs."""

    def __init__(
        self,
        custom_model_pricing: Optional[Dict[str, ModelPricing]] = None,
        custom_tool_pricing: Optional[Dict[str, float]] = None,
    ):
        self._model_catalog: Dict[str, ModelPricing] = dict(DEFAULT_MODEL_PRICING)
        if custom_model_pricing:
            self._model_catalog.update(custom_model_pricing)

        self._tool_catalog: Dict[str, float] = dict(DEFAULT_TOOL_PRICING)
        if custom_tool_pricing:
            self._tool_catalog.update(custom_tool_pricing)

    def register_model_pricing(self, pricing: ModelPricing) -> None:
        """Register or override pricing for a specific model."""
        pricing.is_custom = True
        self._model_catalog[pricing.model_id.lower()] = pricing

    def register_tool_pricing(self, tool_name: str, cost_per_call: float) -> None:
        """Register or override pricing for a specific tool API."""
        self._tool_catalog[tool_name.lower()] = float(cost_per_call)

    def get_model_pricing(self, model_name: Optional[str] = None, provider: Optional[str] = None) -> ModelPricing:
        """Retrieve model pricing with fuzzy alias matching and provider inference."""
        if not model_name:
            return FALLBACK_MODEL_PRICING

        norm_name = str(model_name).strip().lower()

        # 1. Exact match
        if norm_name in self._model_catalog:
            return self._model_catalog[norm_name]

        # 2. Normalized clean name (remove dates like -20241022, version extensions, provider prefixes)
        clean_name = re.sub(r"-\d{8}$", "", norm_name)
        clean_name = re.sub(r"^models/", "", clean_name)
        clean_name = re.sub(r"^openai/", "", clean_name)
        clean_name = re.sub(r"^anthropic/", "", clean_name)
        clean_name = re.sub(r"^google/", "", clean_name)

        if clean_name in self._model_catalog:
            return self._model_catalog[clean_name]

        # 3. Fuzzy partial matching
        for key, pricing in self._model_catalog.items():
            if key in clean_name or clean_name in key:
                return pricing

        # 4. Infer provider fallback
        inferred_provider = provider or self._infer_provider(norm_name)
        if "gpt-4" in norm_name or "o1" in norm_name or "o3" in norm_name or "claude-3-5-sonnet" in norm_name:
            return ModelPricing(norm_name, inferred_provider, 2.50, 10.00, display_name=f"{norm_name} (Auto-Inferred)")
        elif "flash" in norm_name or "mini" in norm_name or "haiku" in norm_name or "8b" in norm_name:
            return ModelPricing(norm_name, inferred_provider, 0.15, 0.60, display_name=f"{norm_name} (Auto-Inferred)")

        return ModelPricing(
            norm_name,
            inferred_provider,
            FALLBACK_MODEL_PRICING.input_price_per_mtoken,
            FALLBACK_MODEL_PRICING.output_price_per_mtoken,
            display_name=f"{norm_name} (Standard Estimate)"
        )

    def _infer_provider(self, model_name: str) -> str:
        """Heuristically infer the provider from model name."""
        m = model_name.lower()
        if any(x in m for x in ["gpt", "text-embedding", "o1", "o3", "davinci"]):
            return "openai"
        if "claude" in m:
            return "anthropic"
        if "gemini" in m or "palm" in m:
            return "google"
        if "llama" in m or "meta" in m:
            return "meta"
        if "deepseek" in m:
            return "deepseek"
        if "mistral" in m or "codestral" in m:
            return "mistral"
        if "command" in m:
            return "cohere"
        return "unknown"

    def calculate_llm_cost(
        self,
        model: Optional[str],
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        provider: Optional[str] = None,
        actual_cost: Optional[float] = None
    ) -> CostBreakdown:
        """Calculate LLM cost for given token usage and model."""
        pricing = self.get_model_pricing(model, provider=provider)
        estimated_cost = pricing.calculate_cost(input_tokens, output_tokens, cached_tokens=cached_tokens)

        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached_tokens,
            model=pricing.model_id,
            provider=pricing.provider,
            input_token_price_per_mtoken=pricing.input_price_per_mtoken,
            output_token_price_per_mtoken=pricing.output_price_per_mtoken,
            estimated_llm_cost=round(estimated_cost, 6),
            tool_api_cost=0.0,
            actual_cost=actual_cost,
        )

    def calculate_tool_cost(
        self,
        tool_name: str,
        call_count: int = 1,
        custom_price: Optional[float] = None
    ) -> float:
        """Calculate total execution cost for external tool calls."""
        if custom_price is not None:
            price = float(custom_price)
        else:
            price = self._tool_catalog.get(tool_name.lower(), 0.0)
        return round(price * max(0, call_count), 6)

    def calculate_trace_cost(
        self,
        trace: Any,
        quality_score: Optional[float] = None,
        default_model: Optional[str] = None,
        actual_cost: Optional[float] = None
    ) -> CostBreakdown:
        """Calculate comprehensive LLM + Tool API cost across an entire execution trace."""
        if not trace:
            return CostBreakdown()

        # Check if actual cost was provided in trace metadata/attributes
        extracted_actual_cost = actual_cost
        if extracted_actual_cost is None and hasattr(trace, "metadata") and isinstance(trace.metadata, dict):
            extracted_actual_cost = trace.metadata.get("actual_cost") or trace.metadata.get("actual_provider_cost")

        total_input_tokens = 0
        total_output_tokens = 0
        cached_tokens = 0
        detected_model = default_model or "gpt-4o"
        detected_provider = "unknown"

        tool_breakdown: Dict[str, Dict[str, Any]] = {}
        total_tool_cost = 0.0
        total_tool_calls = 0

        # Scan spans if present
        spans = getattr(trace, "spans", [])
        for span in spans:
            st = str(getattr(span, "span_type", getattr(span, "step_type", ""))).lower()
            name = getattr(span, "name", getattr(span, "operation_name", "")) or ""
            tool_name = getattr(span, "tool_name", None) or (name if "tool" in st else None)

            # Check for actual cost at span level
            if extracted_actual_cost is None:
                span_attrs = getattr(span, "attributes", {}) or {}
                if isinstance(span_attrs, dict) and "actual_cost" in span_attrs:
                    extracted_actual_cost = float(span_attrs["actual_cost"])

            # 1. LLM Call Span
            if "llm" in st or getattr(span, "input_tokens", 0) > 0 or getattr(span, "output_tokens", 0) > 0:
                in_tok = int(getattr(span, "input_tokens", 0) or 0)
                out_tok = int(getattr(span, "output_tokens", 0) or 0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                span_model = getattr(span, "model", None)
                if span_model:
                    detected_model = span_model

            # 2. Tool Call Span
            if "tool" in st or tool_name:
                t_name = str(tool_name or name).lower()
                total_tool_calls += 1
                if t_name not in tool_breakdown:
                    cost_per_call = self._tool_catalog.get(t_name, 0.0)
                    tool_breakdown[t_name] = {"calls": 0, "cost_per_call": cost_per_call, "total_cost": 0.0}
                tool_breakdown[t_name]["calls"] += 1
                tool_breakdown[t_name]["total_cost"] = round(
                    tool_breakdown[t_name]["calls"] * tool_breakdown[t_name]["cost_per_call"], 6
                )

        # Fallback to trace level token counts if spans had 0
        if total_input_tokens == 0 and total_output_tokens == 0:
            total_input_tokens = getattr(trace, "total_input_tokens", 0) or 0
            total_output_tokens = getattr(trace, "total_output_tokens", 0) or 0

        # Calculate tool cost sum
        total_tool_cost = sum(t["total_cost"] for t in tool_breakdown.values())

        # Pricing lookup
        pricing = self.get_model_pricing(detected_model)
        detected_provider = pricing.provider
        estimated_llm_cost = pricing.calculate_cost(total_input_tokens, total_output_tokens, cached_tokens=cached_tokens)

        # Calculate cost efficiency if quality score provided
        eff = 0.0
        final_total_cost = extracted_actual_cost if extracted_actual_cost is not None else (estimated_llm_cost + total_tool_cost)
        if quality_score is not None and quality_score > 0 and final_total_cost > 0:
            eff = float(quality_score / final_total_cost)

        return CostBreakdown(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            cached_tokens=cached_tokens,
            model=pricing.model_id,
            provider=detected_provider,
            input_token_price_per_mtoken=pricing.input_price_per_mtoken,
            output_token_price_per_mtoken=pricing.output_price_per_mtoken,
            estimated_llm_cost=round(estimated_llm_cost, 6),
            tool_api_cost=round(total_tool_cost, 6),
            total_cost=round(final_total_cost, 6),
            actual_cost=extracted_actual_cost,
            cost_efficiency=eff,
            tool_calls_count=total_tool_calls,
            tool_breakdown=tool_breakdown,
        )

    def calculate_cost_efficiency(self, quality_score: float, cost_usd: float) -> float:
        """Calculate quality score points per USD spent (quality_score / estimated_cost)."""
        if cost_usd <= 0:
            return 0.0
        return round(float(quality_score / cost_usd), 2)


# Global default singleton instance
global_cost_calculator = CostCalculator()
