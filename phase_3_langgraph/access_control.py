from dataclasses import dataclass
import hmac
from typing import Literal

from phase_3_langgraph.employee_ingest import get_privacy_secret
from phase_3_langgraph.pii_guardrails import tokenize_identifier


Role = Literal["employee", "manager", "hr_admin"]
VALID_ROLES = frozenset({"employee", "manager", "hr_admin"})


@dataclass(frozen=True)
class AccessContext:
    """Verified requester identity used for authorization decisions."""

    requester_token: str
    role: Role


def create_access_context(
    authenticated_employee_code: str,
    role: Role,
) -> AccessContext:
    """Create an access context without retaining the raw employee code."""

    if role not in VALID_ROLES:
        raise ValueError("Unsupported access role")

    requester_token = tokenize_identifier(
        authenticated_employee_code,
        get_privacy_secret(),
    )

    return AccessContext(
        requester_token=requester_token,
        role=role,
    )


def authorize_employee_access(
    access_context: AccessContext,
    subject_employee_code: str,
) -> str:
    """Authorize employee self-access and return the approved subject token."""

    subject_token = tokenize_identifier(
        subject_employee_code,
        get_privacy_secret(),
    )

    is_self_access = hmac.compare_digest(
        access_context.requester_token,
        subject_token,
    )

    if access_context.role == "employee" and is_self_access:
        return subject_token

    raise PermissionError(
        "Access to the requested employee record is not permitted"
    )


if __name__ == "__main__":
    employee_context = create_access_context("EMP001", "employee")

    approved_token = authorize_employee_access(
        employee_context,
        "EMP001",
    )
    assert hmac.compare_digest(
        approved_token,
        employee_context.requester_token,
    )
    print("Self-access test passed")

    try:
        authorize_employee_access(employee_context, "EMP002")
    except PermissionError:
        print("Cross-employee access test passed")
    else:
        raise AssertionError("Cross-employee access was not rejected")

    manager_context = create_access_context("EMP001", "manager")

    try:
        authorize_employee_access(manager_context, "EMP001")
    except PermissionError:
        print("Default-deny manager test passed")
    else:
        raise AssertionError("Manager access was not denied by default")

    try:
        create_access_context("EMP001", "contractor")  # type: ignore[arg-type]
    except ValueError:
        print("Invalid-role test passed")
    else:
        raise AssertionError("Invalid role was not rejected")