import csv
import hmac

from phase_3_langgraph.access_control import (
    AccessContext,
    authorize_employee_access,
)
from phase_3_langgraph.employee_ingest import (
    CSV_PATH,
    get_privacy_secret,
)
from phase_3_langgraph.pii_guardrails import (
    tokenize_identifier,
    validate_schema,
)


AUTHORIZED_SELF_FIELDS = {
    "Department": "department",
    "Designation": "designation",
    "Payroll Month": "payroll_month",
    "source_sick_leave_days": "sick_leave_days_in_payroll_month",
    "source_unpaid_leave_days": "unpaid_leave_days_in_payroll_month",
}


def build_authorized_employee_context(
    access_context: AccessContext,
    subject_employee_code: str,
) -> dict[str, str]:
    """Return a minimal employee context after successful self-authorization."""

    approved_subject_token = authorize_employee_access(
        access_context,
        subject_employee_code,
    )
    secret = get_privacy_secret()
    matched_context: dict[str, str] | None = None

    with CSV_PATH.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("Employee CSV has no header")

        validate_schema(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Employee CSV row {row_number} is malformed"
                )

            row_token = tokenize_identifier(
                row["Employee Code"],
                secret,
            )

            if not hmac.compare_digest(row_token, approved_subject_token):
                continue

            if matched_context is not None:
                raise ValueError("Duplicate authorized employee record found")

            matched_context = {
                safe_name: row[source_column]
                for source_column, safe_name in AUTHORIZED_SELF_FIELDS.items()
            }

    if matched_context is None:
        raise LookupError("Authorized employee record was not found")

    return matched_context
