from dataclasses import dataclass
from typing import Literal
from phase_3_langgraph.ingress_guardrails import redact_ingress_text

EgressAction = Literal["allow", "redact", "block"]

HIGH_RISK_CATEGORIES = frozenset(
    {
        "pan",
        "aadhaar_like",
        "bank_account",
        "secret",
    }
)

BLOCKED_RESPONSE = (
    "I’m unable to return that response because it may contain "
    "sensitive personal information. Please contact HR securely."
)

@dataclass(frozen=True)
class EgressGuardResult:
    text: str
    action: EgressAction
    detected_categories: tuple[str, ...]
    redaction_count: int

def guard_egress_text(answer: str) -> EgressGuardResult:
    """Inspect a model answer before returning it to the user."""

    # Empty model output is unsafe/unusable, so block it.
    if not answer or not answer.strip():
        return EgressGuardResult(
            text=BLOCKED_RESPONSE,
            action="block",
            detected_categories=(),
            redaction_count=0,
        )

    # Reuse the PII detector and obtain a sanitized version.
    redaction_result = redact_ingress_text(answer)

    detected_categories = set(
        redaction_result.detected_categories
    )

    # Block the complete answer when high-risk data is detected.
    if detected_categories & HIGH_RISK_CATEGORIES:
        return EgressGuardResult(
            text=BLOCKED_RESPONSE,
            action="block",
            detected_categories=redaction_result.detected_categories,
            redaction_count=redaction_result.redaction_count,
        )

    # Return a redacted answer for lower-risk PII such as email or phone.
    if redaction_result.redaction_count > 0:
        return EgressGuardResult(
            text=redaction_result.text,
            action="redact",
            detected_categories=redaction_result.detected_categories,
            redaction_count=redaction_result.redaction_count,
        )

    # No sensitive information was found.
    return EgressGuardResult(
        text=redaction_result.text,
        action="allow",
        detected_categories=(),
        redaction_count=0,
    )
