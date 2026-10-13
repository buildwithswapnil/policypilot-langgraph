import csv
import os
from pathlib import Path

from dotenv import load_dotenv

from phase_3_langgraph.pii_guardrails import (
    ALLOW,
    COLUMN_RULES,
    sanitize_employee_row,
    validate_schema,
)


MODEL_SAFE_FIELDS = {
    column
    for column, action in COLUMN_RULES.items()
    if action == ALLOW
}

MODEL_SAFE_FIELDS.add("Employee Token")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "employees"
    / "raw"
    / "employees_synthetic.csv"
)
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def get_privacy_secret() -> str:
    secret = os.getenv("PRIVACY_HMAC_SECRET")

    if not secret or not secret.strip():
        raise RuntimeError(
            "PRIVACY_HMAC_SECRET is missing or empty in .env"
        )

    return secret.strip()


def load_sanitized_employees(
    csv_path: Path = CSV_PATH,
) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Employee CSV was not found: {csv_path}"
        )

    secret = get_privacy_secret()
    safe_employees: list[dict[str, str]] = []

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        validate_schema(reader.fieldnames)

        for row in reader:
            safe_row = sanitize_employee_row(row, secret)
            safe_employees.append(safe_row)

    if not safe_employees:
        raise ValueError(
            "Employee CSV contains no employee records"
        )

    return safe_employees


def validate_sanitized_employees(
    employees: list[dict[str, str]],
) -> None:
    """Verify that rows contain only approved model-safe fields."""

    if not employees:
        raise ValueError(
            "No sanitized employee records were provided"
        )

    for row_number, employee in enumerate(
        employees,
        start=1,
    ):
        actual_fields = set(employee)

        missing = MODEL_SAFE_FIELDS - actual_fields
        unexpected = actual_fields - MODEL_SAFE_FIELDS

        if missing or unexpected:
            raise ValueError(
                f"Sanitized employee row {row_number} "
                "has invalid fields. "
                f"Missing: {sorted(missing)}, "
                f"Unexpected: {sorted(unexpected)}"
            )

        employee_token = employee.get(
            "Employee Token",
            "",
        )

        if not employee_token.startswith("employee_"):
            raise ValueError(
                f"Sanitized employee row {row_number} "
                "has an invalid token"
            )


if __name__ == "__main__":
    employees = load_sanitized_employees()
    validate_sanitized_employees(employees)

    print(
        "Sanitized employee count:",
        len(employees),
    )
    print(
        "Safe output fields:",
        sorted(employees[0]),
    )
    print("PII leakage validation passed")