"""
Model and Tool Pricing Catalog for Cost Observability.

Defines pricing per 1 Million tokens (input/output/cached) across major LLM providers
(OpenAI, Anthropic, Google Gemini, Meta LLaMA, DeepSeek, Mistral, Cohere),
plus standard tool/API per-call pricing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ModelPricing:
    """Pricing definition for a specific LLM model."""
    model_id: str
    provider: str
    input_price_per_mtoken: float          # USD per 1M input tokens
    output_price_per_mtoken: float         # USD per 1M output tokens
    cached_input_price_per_mtoken: float = 0.0  # USD per 1M cached prompt tokens (if supported)
    context_window: int = 128_000          # Maximum context window size
    display_name: str = ""
    is_custom: bool = False

    def __post_init__(self):
        if not self.display_name:
            self.display_name = f"{self.provider.capitalize()} {self.model_id}"

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0
    ) -> float:
        """Calculate LLM generation cost in USD for given token counts."""
        uncached_input = max(0, input_tokens - cached_tokens)
        input_cost = (uncached_input / 1_000_000.0) * self.input_price_per_mtoken
        cached_cost = (cached_tokens / 1_000_000.0) * (
            self.cached_input_price_per_mtoken or (self.input_price_per_mtoken * 0.5)
        )
        output_cost = (output_tokens / 1_000_000.0) * self.output_price_per_mtoken
        return input_cost + cached_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "input_price_per_mtoken": self.input_price_per_mtoken,
            "output_price_per_mtoken": self.output_price_per_mtoken,
            "cached_input_price_per_mtoken": self.cached_input_price_per_mtoken,
            "context_window": self.context_window,
            "is_custom": self.is_custom,
        }


# Standard Default LLM Pricing Catalog ($ USD per 1M tokens)
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    # --- OpenAI ---
    "gpt-4o": ModelPricing("gpt-4o", "openai", 2.50, 10.00, cached_input_price_per_mtoken=1.25, context_window=128000, display_name="GPT-4o"),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", "openai", 0.15, 0.60, cached_input_price_per_mtoken=0.075, context_window=128000, display_name="GPT-4o mini"),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", "openai", 10.00, 30.00, context_window=128000, display_name="GPT-4 Turbo"),
    "gpt-4": ModelPricing("gpt-4", "openai", 30.00, 60.00, context_window=8192, display_name="GPT-4"),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", "openai", 0.50, 1.50, context_window=16385, display_name="GPT-3.5 Turbo"),
    "o1": ModelPricing("o1", "openai", 15.00, 60.00, cached_input_price_per_mtoken=7.50, context_window=200000, display_name="OpenAI o1"),
    "o1-mini": ModelPricing("o1-mini", "openai", 3.00, 12.00, cached_input_price_per_mtoken=1.50, context_window=128000, display_name="OpenAI o1-mini"),
    "o3-mini": ModelPricing("o3-mini", "openai", 1.10, 4.40, cached_input_price_per_mtoken=0.55, context_window=200000, display_name="OpenAI o3-mini"),

    # --- Anthropic ---
    "claude-3-5-sonnet": ModelPricing("claude-3-5-sonnet", "anthropic", 3.00, 15.00, cached_input_price_per_mtoken=0.30, context_window=200000, display_name="Claude 3.5 Sonnet"),
    "claude-3-5-haiku": ModelPricing("claude-3-5-haiku", "anthropic", 0.80, 4.00, cached_input_price_per_mtoken=0.08, context_window=200000, display_name="Claude 3.5 Haiku"),
    "claude-3-opus": ModelPricing("claude-3-opus", "anthropic", 15.00, 75.00, cached_input_price_per_mtoken=1.50, context_window=200000, display_name="Claude 3 Opus"),
    "claude-3-haiku": ModelPricing("claude-3-haiku", "anthropic", 0.25, 1.25, cached_input_price_per_mtoken=0.025, context_window=200000, display_name="Claude 3 Haiku"),

    # --- Google Gemini ---
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", "google", 1.25, 5.00, cached_input_price_per_mtoken=0.3125, context_window=2000000, display_name="Gemini 1.5 Pro"),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", "google", 0.075, 0.30, cached_input_price_per_mtoken=0.01875, context_window=1000000, display_name="Gemini 1.5 Flash"),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", "google", 0.10, 0.40, cached_input_price_per_mtoken=0.025, context_window=1000000, display_name="Gemini 2.0 Flash"),
    "gemini-2.0-pro": ModelPricing("gemini-2.0-pro", "google", 1.50, 6.00, cached_input_price_per_mtoken=0.375, context_window=2000000, display_name="Gemini 2.0 Pro"),

    # --- Meta LLaMA / Open Source ---
    "llama-3.3-70b": ModelPricing("llama-3.3-70b", "meta", 0.59, 0.79, context_window=128000, display_name="Llama 3.3 70B"),
    "llama-3.1-70b": ModelPricing("llama-3.1-70b", "meta", 0.59, 0.79, context_window=128000, display_name="Llama 3.1 70B"),
    "llama-3.1-8b": ModelPricing("llama-3.1-8b", "meta", 0.055, 0.055, context_window=128000, display_name="Llama 3.1 8B"),
    "llama-3.1-405b": ModelPricing("llama-3.1-405b", "meta", 3.00, 3.00, context_window=128000, display_name="Llama 3.1 405B"),

    # --- DeepSeek ---
    "deepseek-chat": ModelPricing("deepseek-chat", "deepseek", 0.14, 0.28, cached_input_price_per_mtoken=0.014, context_window=64000, display_name="DeepSeek V3 (Chat)"),
    "deepseek-reasoner": ModelPricing("deepseek-reasoner", "deepseek", 0.55, 2.19, cached_input_price_per_mtoken=0.14, context_window=64000, display_name="DeepSeek R1 (Reasoner)"),

    # --- Mistral ---
    "mistral-large": ModelPricing("mistral-large", "mistral", 2.00, 6.00, context_window=128000, display_name="Mistral Large"),
    "mistral-small": ModelPricing("mistral-small", "mistral", 0.20, 0.60, context_window=128000, display_name="Mistral Small"),
    "codestral": ModelPricing("codestral", "mistral", 0.30, 0.90, context_window=256000, display_name="Codestral"),

    # --- Cohere ---
    "command-r-plus": ModelPricing("command-r-plus", "cohere", 2.50, 10.00, context_window=128000, display_name="Command R+"),
    "command-r": ModelPricing("command-r", "cohere", 0.15, 0.60, context_window=128000, display_name="Command R"),
}

# Fallback generic default pricing if unknown model
FALLBACK_MODEL_PRICING = ModelPricing("default_generic", "unknown", 1.00, 3.00, display_name="Generic LLM (Fallback)")

# Standard Tool / API Per-Call Pricing ($ USD per execution)
DEFAULT_TOOL_PRICING: Dict[str, float] = {
    "web_search": 0.005,
    "google_search": 0.005,
    "serpapi": 0.01,
    "tavily_search": 0.008,
    "brave_search": 0.003,
    "wolfram_alpha": 0.02,
    "code_interpreter": 0.002,
    "database_query": 0.0,
    "calculator": 0.0,
    "search_knowledge_base": 0.0,
    "get_current_date": 0.0,
}
