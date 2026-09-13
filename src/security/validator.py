"""Input, File Upload, and Prompt Injection Security Validators."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


class SecurityValidationError(Exception):
    """Raised when an input, upload, or parameter violates security constraints."""
    pass


@dataclass
class InjectionCheckResult:
    """Outcome of a prompt injection and malicious payload scan."""
    is_suspicious: bool
    risk_score: float                  # 0.0 (clean) to 1.0 (confirmed malicious payload)
    detected_patterns: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    rejection_reason: Optional[str] = None


class SecurityInputValidator:
    """Validates user inputs, file uploads, and scans for prompt injection and malicious content."""

    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024     # 25 MB max compressed archive
    MAX_UNCOMPRESSED_BYTES: int = 50 * 1024 * 1024   # 50 MB max uncompressed
    MAX_FILE_COUNT: int = 500                        # 500 max files
    MAX_TEXT_INPUT_CHARS: int = 50_000               # 50k chars max query input

    BLOCKED_EXTENSIONS = {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".com", ".scr",
        ".msi", ".bat", ".cmd", ".ps1", ".vbs", ".sh", ".bash",
    }

    # Prompt Injection Heuristic Patterns
    PROMPT_INJECTION_SIGNATURES: List[Tuple[str, re.Pattern, float]] = [
        (
            "SYSTEM_OVERRIDE",
            re.compile(r"(?i)(?:ignore\s+(?:all\s+)?previous\s+instructions|disregard\s+(?:all\s+)?prior\s+prompts)"),
            0.9,
        ),
        (
            "ROLEPLAY_JAILBREAK",
            re.compile(r"(?i)(?:you\s+are\s+now\s+(?:in\s+developer\s+mode|unfiltered|DAN|an\s+unrestricted\s+AI))"),
            0.85,
        ),
        (
            "CREDENTIAL_EXTRACTION",
            re.compile(r"(?i)(?:print|reveal|output|display|show)\s+(?:the\s+)?(?:system\s+prompt|api\s+key|environment\s+variables|openai_api_key)"),
            0.8,
        ),
        (
            "SQL_INJECTION_KEYWORD",
            re.compile(r"(?i)\b(?:UNION\s+ALL\s+SELECT|DROP\s+TABLE|ALTER\s+TABLE|;\s*SHUTDOWN)\b"),
            0.9,
        ),
        (
            "HTML_XSS_TAG",
            re.compile(r"<\s*script[^>]*>[\s\S]*?<\s*/\s*script\s*>", re.IGNORECASE),
            0.95,
        ),
        (
            "EVENT_HANDLER_XSS",
            re.compile(r"(?i)\b(?:onerror|onload|onclick|onmouseover)\s*="),
            0.85,
        ),
    ]

    @classmethod
    def validate_upload_archive_size(cls, file_bytes_or_path: Union[bytes, str, Path]) -> int:
        """Validate that an uploaded file archive does not exceed the maximum upload threshold."""
        if isinstance(file_bytes_or_path, (str, Path)):
            p = Path(file_bytes_or_path).resolve()
            if not p.is_file():
                raise SecurityValidationError(f"File not found: '{file_bytes_or_path}'")
            size = p.stat().st_size
        elif isinstance(file_bytes_or_path, bytes):
            size = len(file_bytes_or_path)
        elif hasattr(file_bytes_or_path, "getbuffer"):
            size = len(file_bytes_or_path.getbuffer())
        elif hasattr(file_bytes_or_path, "size"):
            size = file_bytes_or_path.size
        else:
            size = 0

        if size > cls.MAX_UPLOAD_SIZE_BYTES:
            raise SecurityValidationError(
                f"Upload size ({size / (1024*1024):.1f} MB) exceeds maximum allowed limit ({cls.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.1f} MB)."
            )
        return size

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename against path traversal (../), null bytes, and dangerous characters."""
        if not filename:
            return "unnamed_file"

        # Remove null bytes
        cleaned = filename.replace("\0", "")
        # Get pure basename
        basename = os.path.basename(cleaned)
        # Strip path traversal dots
        basename = re.sub(r"\.\.+[/\\ ]*", "", basename)
        # Remove dangerous control chars
        basename = re.sub(r'[\<\>:"/\\|?*]', "_", basename).strip()

        ext = Path(basename).suffix.lower()
        if ext in cls.BLOCKED_EXTENSIONS:
            raise SecurityValidationError(f"Prohibited executable file extension: '{ext}'")

        return basename or "unnamed_file"

    @classmethod
    def validate_text_input(cls, text: str, field_name: str = "Input", max_len: Optional[int] = None) -> str:
        """Validate input length and check for null byte injection."""
        if not text:
            return ""

        max_limit = max_len or cls.MAX_TEXT_INPUT_CHARS
        if len(text) > max_limit:
            raise SecurityValidationError(
                f"{field_name} length ({len(text):,} chars) exceeds maximum allowed threshold ({max_limit:,} chars)."
            )

        if "\0" in text:
            raise SecurityValidationError(f"Null byte detected in {field_name}.")

        return text

    @classmethod
    def scan_prompt_injection(cls, text: str) -> InjectionCheckResult:
        """Scan input query or tool output for prompt injection signatures and XSS scripts."""
        if not text:
            return InjectionCheckResult(is_suspicious=False, risk_score=0.0, sanitized_text="")

        detected: List[str] = []
        max_score = 0.0

        for rule_name, pattern, score in cls.PROMPT_INJECTION_SIGNATURES:
            if pattern.search(text):
                detected.append(rule_name)
                if score > max_score:
                    max_score = score

        # Clean XSS tags if detected
        sanitized = re.sub(r"<\s*script[^>]*>[\s\S]*?<\s*/\s*script\s*>", "[REMOVED_SCRIPT]", text, flags=re.IGNORECASE)

        is_suspicious = max_score >= 0.8
        rejection_reason = f"Potential prompt injection / malicious pattern detected: {', '.join(detected)}" if is_suspicious else None

        return InjectionCheckResult(
            is_suspicious=is_suspicious,
            risk_score=max_score,
            detected_patterns=detected,
            sanitized_text=sanitized,
            rejection_reason=rejection_reason,
        )
