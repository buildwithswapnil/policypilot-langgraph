from phase_3_langgraph.employee_ingest import (
    get_privacy_secret,
    load_sanitized_employees,
    validate_sanitized_employees,
)
from phase_3_langgraph.pii_guardrails import tokenize_identifier


EmployeeRecord = dict[str, str]
EmployeeIndex = dict[str, EmployeeRecord]


def build_employee_index(
    employees: list[EmployeeRecord],
) -> EmployeeIndex:
    """Index sanitized employee records by pseudonymous employee token."""

    validate_sanitized_employees(employees)
    employee_index: EmployeeIndex = {}

    for employee in employees:
        employee_token = employee["Employee Token"]

        if employee_token in employee_index:
            raise ValueError("Duplicate employee token found")

        employee_index[employee_token] = dict(employee)

    return employee_index


def find_employee_by_code(
    employee_code: str,
    employee_index: EmployeeIndex,
) -> EmployeeRecord | None:
    """Find a sanitized employee without storing or comparing the raw code."""

    lookup_token = tokenize_identifier(
        employee_code,
        get_privacy_secret(),
    )
    employee = employee_index.get(lookup_token)

    return dict(employee) if employee is not None else None


if __name__ == "__main__":
    sanitized_employees = load_sanitized_employees()
    index = build_employee_index(sanitized_employees)

    known_employee = find_employee_by_code("EMP001", index)
    unknown_employee = find_employee_by_code("EMP999", index)

    print("Indexed employee count:", len(index))
    print("Known employee found:", known_employee is not None)
    print("Unknown employee rejected:", unknown_employee is None)