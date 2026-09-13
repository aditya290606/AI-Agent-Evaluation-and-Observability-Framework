"""
Safe serialization and credential scrubbing engine for AgentPulse SDK.

Ensures zero secret leakage and crashes during telemetry serialization.
Redacts API keys, bearer tokens, passwords, private keys, and sensitive fields.
"""

import json
import re
from typing import Any, Dict, List, Tuple


# Regex patterns for credential redacting
_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # Anthropic API Keys (sk-ant-...)
    (
        "ANTHROPIC_KEY",
        re.compile(r"\b(sk-ant-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
        "[REDACTED_ANTHROPIC_KEY]",
    ),
    # OpenAI API Keys (sk-...)
    (
        "OPENAI_KEY",
        re.compile(r"\b(sk-(?:proj-|admin-)?[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
        "[REDACTED_OPENAI_KEY]",
    ),
    # Google AI / Gemini Keys
    (
        "GOOGLE_KEY",
        re.compile(r"\b(AIzaSy[a-zA-Z0-9_\-]{33})\b"),
        "[REDACTED_GOOGLE_KEY]",
    ),
    # AWS Access Keys
    (
        "AWS_KEY",
        re.compile(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b"),
        "[REDACTED_AWS_KEY]",
    ),
    # GitHub Tokens
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{50,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    # Generic Bearer Tokens
    (
        "BEARER_TOKEN",
        re.compile(r"(?i)\bBearer\s+([a-zA-Z0-9_\-\.]{20,})\b"),
        "Bearer [REDACTED_TOKEN]",
    ),
    # Passwords in query strings or JSON key-value pairs
    (
        "PASSWORD_FIELD",
        re.compile(r'(?i)(["\']?(?:password|secret|api_key|access_token)["\']?\s*[:=]\s*["\'])([^"\']{3,})(["\'])'),
        r"\1[REDACTED]\3",
    ),
]

_SENSITIVE_KEY_SUBSTRINGS = {
    "password", "secret", "token", "api_key", "apikey",
    "auth", "authorization", "credential", "private_key"
}


def sanitize_text(text: str) -> str:
    """Scrub sensitive credentials and secrets from text."""
    if not isinstance(text, str):
        text = str(text)

    for _, pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text


def sanitize_data(data: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Recursively scrub sensitive keys and credential patterns from structured data."""
    if depth > max_depth:
        return "[TRUNCATED_NESTING]"

    if isinstance(data, str):
        return sanitize_text(data)

    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k)
            k_lower = k_str.lower()
            if any(sub in k_lower for sub in _SENSITIVE_KEY_SUBSTRINGS):
                cleaned[k_str] = "[REDACTED]"
            else:
                cleaned[k_str] = sanitize_data(v, depth + 1, max_depth)
        return cleaned

    if isinstance(data, (list, tuple, set)):
        return [sanitize_data(item, depth + 1, max_depth) for item in data]

    # Primitives
    if isinstance(data, (int, float, bool)) or data is None:
        return data

    return sanitize_text(str(data))


def safe_serialize(obj: Any, max_len: int = 32768) -> str:
    """Safely convert any arbitrary Python object into sanitized JSON or string representation.

    Never raises TypeError or recursion errors.
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        sanitized = sanitize_text(obj)
        return sanitized[:max_len] if len(sanitized) > max_len else sanitized

    # Try JSON serialization
    try:
        cleaned_obj = sanitize_data(obj)

        def _default_encoder(o: Any) -> Any:
            # Handle dataclasses, Pydantic, duck-typed models
            if hasattr(o, "model_dump") and callable(o.model_dump):
                return o.model_dump()
            if hasattr(o, "dict") and callable(o.dict):
                return o.dict()
            if hasattr(o, "__dict__"):
                return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            return str(o)

        serialized = json.dumps(cleaned_obj, default=_default_encoder, ensure_ascii=False)
        return serialized[:max_len] if len(serialized) > max_len else serialized
    except Exception:
        # Fallback to string representation
        text_repr = sanitize_text(str(obj))
        return text_repr[:max_len] if len(text_repr) > max_len else text_repr
