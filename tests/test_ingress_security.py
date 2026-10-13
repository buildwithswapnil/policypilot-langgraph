import unittest

from phase_3_langgraph.access_control import create_access_context
from phase_3_langgraph.employee_context import (
    AUTHORIZED_SELF_FIELDS,
    build_authorized_employee_context,
)
from phase_3_langgraph.ingress_guardrails import redact_ingress_text


class IngressTextRedactionTests(unittest.TestCase):
    def test_redacts_supported_sensitive_patterns(self) -> None:
        raw_values = {
            "person@example.com",
            "+91 98765 43210",
            "ABCDE1234F",
            "1234 5678 9012",
            "account number 123456789012345",
            "sk-exampleSecretValue1234567890",
        }
        question = " | ".join(sorted(raw_values))

        result = redact_ingress_text(question)

        for raw_value in raw_values:
            self.assertNotIn(raw_value, result.text)

        self.assertEqual(result.redaction_count, len(raw_values))
        self.assertEqual(
            set(result.detected_categories),
            {
                "email",
                "phone",
                "pan",
                "aadhaar_like",
                "bank_account",
                "secret",
            },
        )

    def test_preserves_normal_policy_numbers(self) -> None:
        question = "Can I use 10 sick days or submit a ₹10,000 claim?"

        result = redact_ingress_text(question)

        self.assertEqual(result.text, question)
        self.assertEqual(result.redaction_count, 0)

    def test_rejects_blank_question(self) -> None:
        with self.assertRaises(ValueError):
            redact_ingress_text("   ")


class AuthorizedEmployeeContextTests(unittest.TestCase):
    def test_self_access_returns_only_allowlisted_context(self) -> None:
        access_context = create_access_context("EMP001", "employee")

        context = build_authorized_employee_context(
            access_context,
            "EMP001",
        )

        self.assertEqual(
            set(context),
            set(AUTHORIZED_SELF_FIELDS.values()),
        )
        self.assertTrue(
            {
                "First Name",
                "Email",
                "Employee Code",
                "Employee Token",
                "Monthly Salary",
                "Annual Salary",
            }.isdisjoint(context)
        )

    def test_cross_employee_access_is_denied(self) -> None:
        access_context = create_access_context("EMP001", "employee")

        with self.assertRaises(PermissionError):
            build_authorized_employee_context(
                access_context,
                "EMP002",
            )

    def test_missing_authorized_record_is_rejected(self) -> None:
        access_context = create_access_context("EMP999", "employee")

        with self.assertRaises(LookupError):
            build_authorized_employee_context(
                access_context,
                "EMP999",
            )


if __name__ == "__main__":
    unittest.main()
