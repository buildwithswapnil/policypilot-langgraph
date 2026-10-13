from dataclasses import dataclass
import re
from typing import Pattern


@dataclass(frozen=True)
class IngressRedactionResult:
    """Sanitized text plus safe metadata about detected sensitive data."""

    text: str
    detected_categories: tuple[str, ...]
    redaction_count: int


PII_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "secret",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "email",
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pan",
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
    ),
    (
        "bank_account",
        re.compile(
            r"\b(?:bank\s+)?account(?:\s+(?:number|no\.?))?"
            r"(?:\s+is)?\s*[:#-]?\s*\d{9,18}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "aadhaar_like",
        re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\w)"
        ),
    ),
)


def redact_ingress_text(text: str) -> IngressRedactionResult:
    """Redact supported PII patterns before text reaches retrieval or an LLM."""

    if not text or not text.strip():
        raise ValueError("Question cannot be empty")

    sanitized_text = text.strip()
    detected_categories: list[str] = []
    redaction_count = 0

    for category, pattern in PII_PATTERNS:
        replacement = f"[{category.upper()}_REDACTED]"
        sanitized_text, count = pattern.subn(replacement, sanitized_text)

        if count:
            detected_categories.append(category)
            redaction_count += count

    return IngressRedactionResult(
        text=sanitized_text,
        detected_categories=tuple(detected_categories),
        redaction_count=redaction_count,
    )
