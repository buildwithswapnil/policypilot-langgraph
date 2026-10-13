from collections import Counter
import hashlib
import hmac


DROP = "drop"
TOKENIZE = "tokenize"
ALLOW = "allow"
RESTRICT = "restrict"


COLUMN_RULES = {
    "No": DROP,
    "First Name": DROP,
    "Last Name": DROP,
    "Gender": RESTRICT,
    "Start Date": RESTRICT,
    "Years": RESTRICT,
    "Department": ALLOW,
    "Country": ALLOW,
    "Center": ALLOW,
    "Monthly Salary": RESTRICT,
    "Annual Salary": RESTRICT,
    "Job Rate": RESTRICT,
    "Sick Leaves": RESTRICT,
    "Unpaid Leaves": RESTRICT,
    "Overtime Hours": RESTRICT,
    "Employee Code": TOKENIZE,
    "Email": DROP,
    "Designation": ALLOW,
    "Payroll Month": ALLOW,
    "calculation_strategy": ALLOW,
    "salary_divisor": ALLOW,
    "monthly_gross_salary": RESTRICT,
    "basic_salary": RESTRICT,
    "hra": RESTRICT,
    "other_allowances": RESTRICT,
    "overtime_rate_per_hour": RESTRICT,
    "fixed_deductions": RESTRICT,
    "effective_from": RESTRICT,
    "effective_to": RESTRICT,
    "source_sick_leave_days": RESTRICT,
    "source_unpaid_leave_days": RESTRICT,
    "demo_overtime_hours": RESTRICT,
    "timesheet_work_days": ALLOW,
    "timesheet_daily_hours": ALLOW,
}


def validate_schema(headers: list[str]) -> None:
    """Reject CSV schemas that are missing, unexpected, or duplicated."""

    expected = set(COLUMN_RULES)
    received = set(headers)

    missing = expected - received
    unexpected = received - expected

    counts = Counter(headers)
    duplicates = {
        column
        for column, count in counts.items()
        if count > 1
    }

    if missing or unexpected or duplicates:
        raise ValueError(
            "Invalid CSV schema. "
            f"Missing: {sorted(missing)}, "
            f"Unexpected: {sorted(unexpected)}, "
            f"Duplicates: {sorted(duplicates)}"
        )


def tokenize_identifier(value: str, secret: str) -> str:
    """Convert an identifier into a stable pseudonymous token."""

    if not value or not value.strip():
        raise ValueError("Identifier cannot be empty")

    if not secret or not secret.strip():
        raise ValueError("Privacy secret cannot be empty")

    normalized_value = value.strip().upper()

    digest = hmac.new(
        key=secret.encode("utf-8"),
        msg=normalized_value.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return f"employee_{digest[:16]}"


def sanitize_employee_row(
    row: dict[str, str],
    secret: str,
) -> dict[str, str]:
    """Create a model-safe employee record using the configured column rules."""

    validate_schema(list(row))

    safe_row: dict[str, str] = {}

    for column, value in row.items():
        action = COLUMN_RULES[column]

        if action == ALLOW:
            safe_row[column] = value
        elif action == TOKENIZE:
            safe_row["Employee Token"] = tokenize_identifier(value, secret)
        elif action in {DROP, RESTRICT}:
            continue
        else:
            raise ValueError(f"Unknown privacy action for {column!r}: {action!r}")

    return safe_row


if __name__ == "__main__":
    # Schema validation tests
    valid_headers = list(COLUMN_RULES)
    validate_schema(valid_headers)
    print("Valid schema test passed")

    unexpected_headers = list(COLUMN_RULES) + ["Aadhaar Number"]

    try:
        validate_schema(unexpected_headers)
    except ValueError as error:
        print(f"Unexpected-column test passed: {error}")
    else:
        raise AssertionError("Unexpected column was not rejected")

    missing_headers = [
        column
        for column in COLUMN_RULES
        if column != "Email"
    ]

    try:
        validate_schema(missing_headers)
    except ValueError as error:
        print(f"Missing-column test passed: {error}")
    else:
        raise AssertionError("Missing column was not rejected")

    duplicate_headers = list(COLUMN_RULES) + ["Email"]

    try:
        validate_schema(duplicate_headers)
    except ValueError as error:
        print(f"Duplicate-column test passed: {error}")
    else:
        raise AssertionError("Duplicate column was not rejected")

    # Tokenization tests
    test_secret = "local-test-secret-do-not-use-in-production"

    token_one = tokenize_identifier("EMP001", test_secret)
    token_one_again = tokenize_identifier(" emp001 ", test_secret)
    token_two = tokenize_identifier("EMP002", test_secret)
    token_with_other_secret = tokenize_identifier(
        "EMP001",
        "different-local-test-secret",
    )

    assert token_one == token_one_again
    assert token_one != token_two
    assert token_one != token_with_other_secret
    assert "EMP001" not in token_one

    print("Tokenization tests passed")
    print(f"Example safe token: {token_one}")

    try:
        tokenize_identifier("", test_secret)
    except ValueError:
        print("Empty-identifier test passed")
    else:
        raise AssertionError("Empty identifier was not rejected")

    try:
        tokenize_identifier("EMP001", "")
    except ValueError:
        print("Empty-secret test passed")
    else:
        raise AssertionError("Empty secret was not rejected")

    # Row sanitization tests
    test_row = {
        column: f"test-value-for-{column}"
        for column in COLUMN_RULES
    }
    test_row["Employee Code"] = "EMP001"

    safe_row = sanitize_employee_row(test_row, test_secret)

    assert "First Name" not in safe_row
    assert "Email" not in safe_row
    assert "Monthly Salary" not in safe_row
    assert "Sick Leaves" not in safe_row
    assert "Employee Code" not in safe_row

    assert "Department" in safe_row
    assert "Designation" in safe_row
    assert "Employee Token" in safe_row
    assert safe_row["Employee Token"] == token_one

    print("Row sanitization tests passed")
