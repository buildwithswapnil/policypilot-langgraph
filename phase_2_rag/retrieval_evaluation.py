from phase_2_rag.vector_store import MINIMUM_RELEVANCE, get_vector_store


# Each tuple contains: question, expected policy section.
# None means that the handbook does not contain the answer.
TEST_CASES = [
    ("How many casual leave days do I receive?", "Casual Leave Policy"),
    ("Do I need a certificate after three sick days?", "Sick Leave Policy"),
    ("How often can I work from home?", "Work From Home Policy"),
    ("When will my monthly salary be credited?", "Payroll Policy"),
    ("Does a ₹12,000 expense need approval?", "Reimbursement Policy"),
    ("How many parental leave days do I receive?", None),
    ("What is the maternity leave policy?", None),
    ("Can I take more than 6 sick leaves?", "Sick Leave Policy"),
    ("Does the company provide health insurance?", None),
    ("What is the office dress code?", None),
    ("Do employees receive overtime pay?", None),
    ("How many public holidays are provided?", None),
]


vector_store = get_vector_store()

passed = 0
answerable_scores = []
unanswerable_scores = []

print(f"Evaluating retrieval with threshold {MINIMUM_RELEVANCE:.2f}\n")

for number, (question, expected_section) in enumerate(TEST_CASES, start=1):
    matches = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=1,
    )

    if not matches:
        raise RuntimeError(
            "The Chroma collection is empty. Run retriever.py once to index the policy."
        )

    document, score = matches[0]
    retrieved_section = document.metadata["policy_section"]

    if expected_section is None:
        predicted_section = (
            retrieved_section if score >= MINIMUM_RELEVANCE else None
        )
        is_correct = predicted_section is None
        unanswerable_scores.append(score)
    else:
        predicted_section = (
            retrieved_section if score >= MINIMUM_RELEVANCE else None
        )
        is_correct = predicted_section == expected_section

        if retrieved_section == expected_section:
            answerable_scores.append(score)

    passed += int(is_correct)
    status = "PASS" if is_correct else "FAIL"

    print(f"{number:>2}. {status} | score={score:.3f}")
    print(f"    Question:  {question}")
    print(f"    Expected:  {expected_section or 'REJECT'}")
    print(f"    Retrieved: {retrieved_section}")

accuracy = passed / len(TEST_CASES)

print("\n--- Evaluation summary ---")
print(f"Passed:   {passed}/{len(TEST_CASES)}")
print(f"Accuracy: {accuracy:.1%}")

if answerable_scores and unanswerable_scores:
    lowest_answerable = min(answerable_scores)
    highest_unanswerable = max(unanswerable_scores)

    print(f"Lowest correctly retrieved answerable score: {lowest_answerable:.3f}")
    print(f"Highest unanswerable score:                  {highest_unanswerable:.3f}")

    if lowest_answerable > highest_unanswerable:
        suggested_threshold = (lowest_answerable + highest_unanswerable) / 2
        print(f"Suggested experimental threshold:           {suggested_threshold:.3f}")
    else:
        print(
            "No single threshold cleanly separates these test cases. "
            "The retrieval strategy needs improvement."
        )
