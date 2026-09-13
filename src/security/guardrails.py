"""Resource constraint guardrails and execution limits."""

from __future__ import annotations

import time
from typing import Any, List, Optional


class ResourceLimitExceededError(Exception):
    """Raised when an operation exceeds memory, timeout, or payload thresholds."""
    pass


class ResourceGuardrails:
    """Enforces execution timeouts, output buffer caps, and batch evaluation bounds."""

    DEFAULT_OUTPUT_LIMIT_BYTES: int = 500 * 1024       # 500 KB per output span
    DEFAULT_MAX_BATCH_TEST_CASES: int = 2_000          # 2000 cases per single batch eval run
    DEFAULT_MAX_TRACE_SPANS: int = 500                 # 500 spans max per trace

    @classmethod
    def truncate_output(cls, output: Any, max_bytes: Optional[int] = None) -> str:
        """Truncate excessive output strings to prevent memory exhaustion and DoS."""
        if output is None:
            return ""

        text = output if isinstance(output, str) else str(output)
        limit = max_bytes or cls.DEFAULT_OUTPUT_LIMIT_BYTES

        if len(text) > limit:
            return text[:limit] + f"\n... [TRUNCATED: Output exceeded safety cap of {limit // 1024} KB]"
        return text

    @classmethod
    def validate_batch_size(cls, test_cases: List[Any], max_cases: Optional[int] = None) -> int:
        """Ensure evaluation batch size does not exceed memory boundaries."""
        limit = max_cases or cls.DEFAULT_MAX_BATCH_TEST_CASES
        count = len(test_cases)
        if count > limit:
            raise ResourceLimitExceededError(
                f"Evaluation batch size ({count:,} cases) exceeds maximum limit ({limit:,} cases)."
            )
        return count

    @classmethod
    def enforce_batch_limit(cls, items: List[Any], max_items: Optional[int] = None) -> List[Any]:
        """Bound list of items to safe batch processing limits."""
        limit = max_items or cls.DEFAULT_MAX_BATCH_TEST_CASES
        return items[:limit]

