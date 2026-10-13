from dataclasses import dataclass
import re

from phase_2_rag.retriever import answer_question


@dataclass(frozen=True)
class AnswerTestCase:
    question: str
    required_phrase_groups: tuple[tuple[str, ...], ...]
    forbidden_phrases: tuple[str, ...] = ()


TEST_CASES = [
    AnswerTestCase(
        question="How many sick-leave days do full-time employees receive?",
        required_phrase_groups=(
            ("10", "ten"),
            ("sick leave",),
        ),
        forbidden_phrases=("12 sick leave", "unlimited"),
    ),
    AnswerTestCase(
        question="Can I take more than 6 sick leaves?",
        required_phrase_groups=(
            ("10", "ten"),
            ("medical certificate",),
        ),
        forbidden_phrases=("unlimited", "not covered"),
    ),
    AnswerTestCase(
        question="Can I take three consecutive sick-leave days?",
        required_phrase_groups=(("medical certificate",),),
        forbidden_phrases=("manager approval",),
    ),
    AnswerTestCase(
        question="What is the maternity-leave policy?",
        required_phrase_groups=(
            ("not covered", "do not contain", "does not contain"),
            ("contact hr",),
        ),
        forbidden_phrases=(
            "10 days of maternity leave",
            "12 days of maternity leave",
        ),
    ),
    AnswerTestCase(
        question="Can I work from home for three days every week?",
        required_phrase_groups=(
            ("two days", "2 days"),
            ("manager approval",),
        ),
        forbidden_phrases=(
            "three days are allowed",
            "work from home for three days",
        ),
    ),
    AnswerTestCase(
        question="Does a ₹12,000 reimbursement need approval?",
        required_phrase_groups=(("department head approval",),),
        forbidden_phrases=("no approval",),
    ),
]


def normalize(text: str) -> str:
    normalized = text.casefold().replace("-", " ").replace("–", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def evaluate_answer(
    answer: str,
    test_case: AnswerTestCase,
) -> tuple[list[tuple[str, ...]], list[str]]:
    normalized_answer = normalize(answer)

    missing_groups = [
        phrase_group
        for phrase_group in test_case.required_phrase_groups
        if not any(normalize(phrase) in normalized_answer for phrase in phrase_group)
    ]
    forbidden_matches = [
        phrase
        for phrase in test_case.forbidden_phrases
        if normalize(phrase) in normalized_answer
    ]

    return missing_groups, forbidden_matches


def main() -> None:
    passed = 0

    print("Running end-to-end PolicyPilot answer evaluation...\n")

    for number, test_case in enumerate(TEST_CASES, start=1):
        try:
            answer = answer_question(test_case.question)
        except Exception as error:
            print(f"{number:>2}. ERROR")
            print(f"    Question: {test_case.question}")
            print(f"    {type(error).__name__}: {error}\n")
            continue

        missing_groups, forbidden_matches = evaluate_answer(answer, test_case)
        is_correct = not missing_groups and not forbidden_matches
        passed += int(is_correct)
        status = "PASS" if is_correct else "FAIL"

        print(f"{number:>2}. {status}")
        print(f"    Question: {test_case.question}")
        print(f"    Answer:   {answer}")

        if missing_groups:
            readable_groups = [" or ".join(group) for group in missing_groups]
            print(f"    Missing:  {readable_groups}")

        if forbidden_matches:
            print(f"    Forbidden claims found: {forbidden_matches}")

        print()

    accuracy = passed / len(TEST_CASES)

    print("--- Answer evaluation summary ---")
    print(f"Passed:   {passed}/{len(TEST_CASES)}")
    print(f"Accuracy: {accuracy:.1%}")

    if passed != len(TEST_CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
