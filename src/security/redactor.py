"""High-coverage Secret and Credential Redaction Engine.

Detects, masks, and redacts sensitive credentials (OpenAI, Anthropic, Gemini, AWS,
GitHub, HuggingFace, Database URIs, Bearer tokens, private keys, passwords) across
text, nested dictionaries, traces, spans, and Python logs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class SecretRedactor:
    """Centralized secret detection and redaction engine."""

    # Regex patterns for high-risk credentials and API keys
    PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        # Anthropic API Keys (sk-ant-...)
        (
            "ANTHROPIC_KEY",
            re.compile(r"\b(sk-ant-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
            "[REDACTED_ANTHROPIC_KEY]",
        ),
        # OpenAI API Keys (sk-proj-..., sk-admin-..., sk-...)
        (
            "OPENAI_KEY",
            re.compile(r"\b(sk-(?!ant-)(?:proj-|admin-)?[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
            "[REDACTED_OPENAI_KEY]",
        ),
        # Google AI / Gemini API Keys (AIzaSy...)
        (
            "GOOGLE_KEY",
            re.compile(r"\b(AIzaSy[a-zA-Z0-9_\-]{33})\b"),
            "[REDACTED_GOOGLE_KEY]",
        ),
        # AWS Access Key IDs (AKIA..., ASIA...)
        (
            "AWS_ACCESS_KEY",
            re.compile(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b"),
            "[REDACTED_AWS_ACCESS_KEY]",
        ),
        # AWS Secret Keys
        (
            "AWS_SECRET_KEY",
            re.compile(r'(?i)(?:aws_secret_access_key|aws_secret_key|secret_key)\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?'),
            "[REDACTED_AWS_SECRET_KEY]",
        ),
        # GitHub Personal Access Tokens (ghp_..., github_pat_...)
        (
            "GITHUB_TOKEN",
            re.compile(r"\b(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{50,})\b"),
            "[REDACTED_GITHUB_TOKEN]",
        ),
        # HuggingFace Tokens (hf_...)
        (
            "HUGGINGFACE_TOKEN",
            re.compile(r"\b(hf_[a-zA-Z0-9]{34,})\b"),
            "[REDACTED_HF_TOKEN]",
        ),
        # Generic Bearer Tokens
        (
            "BEARER_TOKEN",
            re.compile(r'(?i)\bBearer\s+([a-zA-Z0-9_\-\.]{25,})\b'),
            "Bearer [REDACTED_BEARER_TOKEN]",
        ),
        # Database URIs with credentials (e.g. postgresql://user:pass@host/db)
        (
            "DATABASE_URI",
            re.compile(r"((?:postgres|postgresql|mysql|mssql|mongodb|redis|sqlite)://[^:]+:)([^@]+)(@[^\s/]+)"),
            r"\1[REDACTED_PASSWORD]\3",
        ),
        # RSA / EC / DSA / OPENSSH Private Keys
        (
            "PRIVATE_KEY",
            re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----[\s\S]+?-----END (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----"),
            "[REDACTED_PRIVATE_KEY]",
        ),
        # Key-Value Password/Secret Patterns in JSON or config strings
        (
            "JSON_PASSWORD",
            re.compile(r'(?i)(["\']?(?:password|passwd|secret|api_key|access_token|private_key|auth_token)["\']?\s*[:=]\s*["\'])([^"\']{4,})(["\'])'),
            r"\1[REDACTED_SECRET]\3",
        ),
    ]

    @classmethod
    def redact_text(cls, text: Optional[str]) -> str:
        """Scan string and replace all matched secrets with safe masked placeholders."""
        if not text or not isinstance(text, str):
            return "" if text is None else str(text)

        sanitized = text
        for name, pattern, replacement in cls.PATTERNS:
            if "\\" in replacement:
                # Group substitution
                sanitized = pattern.sub(replacement, sanitized)
            else:
                # Direct match substitution
                sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    @classmethod
    def redact_data(cls, data: Any) -> Any:
        """Recursively redact secrets across dictionaries, lists, and primitives."""
        if data is None:
            return None
        if isinstance(data, str):
            return cls.redact_text(data)
        if isinstance(data, dict):
            sanitized_dict: Dict[str, Any] = {}
            for k, v in data.items():
                # If key itself denotes a secret, redact value entirely
                k_lower = str(k).lower()
                if any(sec_term in k_lower for sec_term in ["password", "passwd", "secret", "api_key", "token", "credential", "auth_token"]):
                    sanitized_dict[k] = "[REDACTED_SECRET]"
                else:
                    sanitized_dict[k] = cls.redact_data(v)
            return sanitized_dict
        if isinstance(data, list):
            return [cls.redact_data(item) for item in data]
        if isinstance(data, tuple):
            return tuple(cls.redact_data(item) for item in data)
        return data

    @classmethod
    def redact_span(cls, span: Any) -> Any:
        """Sanitize a Span or TracedStep object in-place."""
        if hasattr(span, "input_data") and span.input_data:
            span.input_data = cls.redact_data(span.input_data)
        if hasattr(span, "output_data") and span.output_data:
            span.output_data = cls.redact_data(span.output_data)
        if hasattr(span, "error") and span.error:
            span.error = cls.redact_text(str(span.error))
        if hasattr(span, "attributes") and isinstance(span.attributes, dict):
            span.attributes = cls.redact_data(span.attributes)
        elif hasattr(span, "metadata") and isinstance(getattr(span, "metadata", None), dict):
            try:
                span.metadata = cls.redact_data(span.metadata)
            except AttributeError:
                pass
        if hasattr(span, "children") and isinstance(span.children, list):
            for child in span.children:
                cls.redact_span(child)
        return span

    @classmethod
    def redact_trace(cls, trace: Any) -> Any:
        """Sanitize a Trace or RunTrace object in-place."""
        if hasattr(trace, "query") and trace.query:
            trace.query = cls.redact_text(trace.query)
        if hasattr(trace, "final_answer") and trace.final_answer:
            trace.final_answer = cls.redact_text(trace.final_answer)
        if hasattr(trace, "metadata") and isinstance(getattr(trace, "metadata", None), dict):
            try:
                trace.metadata = cls.redact_data(trace.metadata)
            except AttributeError:
                pass
        if hasattr(trace, "spans") and isinstance(trace.spans, list):
            for sp in trace.spans:
                cls.redact_span(sp)
        return trace


class RedactingLoggingFormatter(logging.Formatter):
    """Python standard logging formatter that automatically strips secrets from all log messages."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return SecretRedactor.redact_text(original)


def configure_secure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with SecretRedactor sanitizing formatter."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(RedactingLoggingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
