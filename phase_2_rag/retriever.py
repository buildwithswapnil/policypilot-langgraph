from dataclasses import dataclass
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from phase_2_rag.vector_store import (
    DATABASE_PATH,
    MINIMUM_RELEVANCE,
    get_vector_store,
)


NO_POLICY_MESSAGE = (
    "I could not find relevant information in the available HR policies. "
    "Please contact HR for the applicable policy."
)

SupportStatus = Literal["supported", "partial", "unsupported"]


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="A concise, friendly answer grounded in the policy context."
    )
    support_status: SupportStatus = Field(
        description=(
            "Use 'supported' when the context fully answers the question, "
            "'partial' when it provides relevant facts but HR must confirm an "
            "exception or missing detail, and 'unsupported' when it contains "
            "no applicable policy facts."
        )
    )


@dataclass(frozen=True)
class PolicyAnswer:
    answer: str
    support_status: SupportStatus
    policy_section: str | None
    source: str | None
    relevance_score: float | None

vector_store = get_vector_store()
stored_data = vector_store.get()

if not stored_data["ids"]:
    raise RuntimeError(
        "The policy index is empty. Run: python -m phase_2_rag.ingest"
    )

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are PolicyPilot, an HR policy assistant.

Use only the supplied policy context. Do not invent policy facts.

Follow this decision process:

1. If the context contains a rule applicable to the question, answer using it.
   Applicability includes direct comparisons and simple calculations.

2. When the policy states a maximum, minimum, deadline, or threshold:
   - Compare the employee's requested value with the policy value.
   - If the request exceeds the permitted value, clearly say so.
   - State what the policy does permit.
   - Do not classify the question as unsupported merely because the requested
     value differs from the permitted value.

3. If a general policy may apply but eligibility is not confirmed, give a
   conditional answer beginning with "If" and identify what HR must confirm.

4. Use the unavailable-policy response only when the context is about a
   genuinely different topic and provides no applicable rule.

5. Different leave categories are not interchangeable. Sick, casual,
   maternity, and parental leave are separate topics.

6. Begin with a direct answer and use a friendly, professional tone. Answer in
   no more than three concise sentences. Mention the policy section and any
   applicable requirements. Recommend contacting HR only for exceptions or
   information that the context does not cover.

Examples:

Context:
Employees may work from home for up to two days per week with manager approval.
Requests must be submitted one working day in advance.

Question:
Can I apply for three days of work from home?

Answer:
No. The Work From Home Policy allows up to two days per week with manager
approval. Submit the request at least one working day in advance.

Context:
Business expenses must be submitted within 30 days with a receipt.

Question:
Can I reimburse a team outing?

Answer:
If the team outing was an approved business expense, submit it within 30 days
with a receipt. The Reimbursement Policy does not confirm team-outing
eligibility, so contact HR or Finance to verify it.

Context:
Employees receive ten paid sick-leave days annually.

Question:
Can I take maternity leave?

Answer:
Maternity leave is not covered in the available HR policies. Please contact HR
for the applicable policy.

Context:
Every full-time employee receives 10 days of paid sick leave per calendar year.
More than two consecutive sick-leave days require a medical certificate.

Question:
Can I take more than 6 sick leaves?

Answer:
If you mean sick-leave days, yes. Under the Sick Leave Policy, a full-time
employee can use up to 10 paid sick-leave days per calendar year. If more than
two days are consecutive, you must submit a medical certificate; contact HR
for exceptions or leave beyond the stated entitlement.

Interpret policy rules carefully:

1. A HARD LIMIT uses language such as "up to", "maximum", or "cannot exceed".
   A request above a hard limit is not permitted by the current policy.

2. A CONDITIONAL THRESHOLD uses language such as:
   "more than N requires approval, documentation, or a certificate."
   Exceeding this threshold is not prohibited. It triggers the stated
   requirement.

3. An ENTITLEMENT states an amount for a time period, such as "12 days per
   calendar year." It is not automatically a per-request limit.

4. Before comparing numbers, match their units and time periods:
   - annual days
   - days per week
   - consecutive days
   - reimbursement amount
   Do not compare values with different scopes.

5. Never convert an approval or documentation requirement into a prohibition.

Context:
Employees receive 12 casual-leave days per calendar year. More than three
consecutive casual-leave days require manager approval. Requests must be
submitted through the HR portal.

Question:
Can I take five casual-leave days at once?

Answer:
If you have sufficient leave balance, you may request five consecutive
casual-leave days through the HR portal. Because the request exceeds three
consecutive days, manager approval is required.
""",
        ),
        (
            "human",
            """
Policy context:
{context}

Employee question:
{question}
""",
        ),
    ]
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1,
)
structured_llm = llm.with_structured_output(
    GeneratedAnswer,
    method="json_schema",
)


def answer_question_details(
    question: str,
    debug: bool = False,
) -> PolicyAnswer:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    scored_results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=5,
    )
    results = [
        (document, score)
        for document, score in scored_results
        if score >= MINIMUM_RELEVANCE
    ]
    print(results)
    if debug:
        print("\nChroma database:", DATABASE_PATH)
        print("Stored chunk IDs:", stored_data["ids"])
        print("Total stored chunks:", len(stored_data["ids"]))
        print("\n--- Retrieved policy chunks ---")

    if not results:
        return PolicyAnswer(
            answer=NO_POLICY_MESSAGE,
            support_status="unsupported",
            policy_section=None,
            source=None,
            relevance_score=0.5,
        )

    context = "\n\n".join(
        (
            f"Policy section: {document.metadata['policy_section']}\n"
            f"Source: {document.metadata['source']}\n"
            f"{document.page_content}"
        )
        for document, score in results
    )

    formatted_prompt = rag_prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    if debug:
        print("\n--- Retrieved policy section ---")

        for rank, (document, score) in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print("Relevance score:", round(score, 3))
            print("Policy section:", document.metadata["policy_section"])
            print("Source:", document.metadata["source"])
            print("Content:")
            print(document.page_content)

        print("\n--- Prompt sent to the model ---")

        for message in formatted_prompt.messages:
            print(f"\nMessage type: {message.type}")
            print(message.content)

    generated = structured_llm.invoke(formatted_prompt.messages)

    if not isinstance(generated, GeneratedAnswer):
        raise TypeError("The model returned an invalid structured answer.")

    document, score = results[0]
    has_supporting_source = generated.support_status != "unsupported"

    return PolicyAnswer(
        answer=generated.answer.strip(),
        support_status=generated.support_status,
        policy_section=(
            document.metadata["policy_section"] if has_supporting_source else None
        ),
        source=document.metadata["source"] if has_supporting_source else None,
        relevance_score=score if has_supporting_source else None,
    )


def answer_question(question: str, debug: bool = False) -> str:
    return answer_question_details(question, debug=debug).answer


def main() -> None:
    question = input("\nAsk an HR policy question: ")

    try:
        result = answer_question_details(question)
    except ValueError as error:
        print(f"\nPolicyPilot: {error}")
        return

    print(f"\nPolicyPilot: {result.answer}")
    print(f"Support status: {result.support_status}")

    if result.source:
        print(f"Source: {result.policy_section} ({result.source})")
        print(f"Relevance score: {result.relevance_score:.3f}")


if __name__ == "__main__":
    main()
