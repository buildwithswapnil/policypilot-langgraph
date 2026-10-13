import unittest

from phase_3_langgraph.egress_guardrails import (
    BLOCKED_RESPONSE,
    guard_egress_text,
)


class EgressGuardrailTests(unittest.TestCase):
    def test_allows_clean_answer(self) -> None:
        answer = (
            "The Sick Leave Policy provides "
            "10 paid days per calendar year."
        )

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "allow")
        self.assertEqual(result.text, answer)
        self.assertEqual(result.redaction_count, 0)
        self.assertEqual(result.detected_categories, ())

    def test_redacts_email(self) -> None:
        answer = "Contact person@example.com for assistance."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "redact")
        self.assertNotIn("person@example.com", result.text)
        self.assertIn("[EMAIL_REDACTED]", result.text)
        self.assertIn("email", result.detected_categories)

    def test_redacts_phone(self) -> None:
        answer = "Call +91 98765 43210 for assistance."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "redact")
        self.assertNotIn("+91 98765 43210", result.text)
        self.assertIn("[PHONE_REDACTED]", result.text)

    def test_blocks_pan(self) -> None:
        answer = "The employee PAN is ABCDE1234F."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "block")
        self.assertEqual(result.text, BLOCKED_RESPONSE)
        self.assertNotIn("ABCDE1234F", result.text)
        self.assertIn("pan", result.detected_categories)

    def test_blocks_aadhaar_like_number(self) -> None:
        answer = "The identifier is 1234 5678 9012."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "block")
        self.assertEqual(result.text, BLOCKED_RESPONSE)
        self.assertNotIn("1234 5678 9012", result.text)

    def test_blocks_bank_account(self) -> None:
        answer = "The bank account number is 123456789012345."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "block")
        self.assertEqual(result.text, BLOCKED_RESPONSE)
        self.assertNotIn("123456789012345", result.text)
        self.assertIn("bank_account", result.detected_categories)

    def test_blocks_secret(self) -> None:
        answer = "Use sk-exampleSecretValue1234567890 for access."

        result = guard_egress_text(answer)

        self.assertEqual(result.action, "block")
        self.assertEqual(result.text, BLOCKED_RESPONSE)
        self.assertNotIn("sk-exampleSecretValue1234567890", result.text)
        self.assertIn("secret", result.detected_categories)

    def test_blocks_blank_output(self) -> None:
        result = guard_egress_text("   ")

        self.assertEqual(result.action, "block")
        self.assertEqual(result.text, BLOCKED_RESPONSE)
        self.assertEqual(result.redaction_count, 0)


if __name__ == "__main__":
    unittest.main()
