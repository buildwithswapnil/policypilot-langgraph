from typing import Literal, NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from phase_2_rag.vector_store import (
    MINIMUM_RELEVANCE,
    get_vector_store,
)
from phase_3_langgraph.access_control import AccessContext
from phase_3_langgraph.egress_guardrails import guard_egress_text
from phase_3_langgraph.ingress_guardrails import redact_ingress_text


vector_store = get_vector_store()


QuestionIntent = Literal[
    "policy",
    "employee",
    "combined",
    "out_of_scope",
]


class PolicyState(TypedDict):
    question: str

    ingress_pii_categories: NotRequired[tuple[str, ...]]
    ingress_redaction_count: NotRequired[int]

    normalized_question: NotRequired[str]

    intent: NotRequired[QuestionIntent]
    intent_reason: NotRequired[str]

    access_context: NotRequired[AccessContext]
    employee_context: NotRequired[dict[str, str]]

    is_policy_question: NotRequired[bool]

    evidence_found: NotRequired[bool]
    evidence_applicable: NotRequired[bool]
    evidence_reason: NotRequired[str]

    policy_section: NotRequired[str]
    source: NotRequired[str]
    relevance_score: NotRequired[float]
    context: NotRequired[str]

    answer: NotRequired[str]

    egress_action: NotRequired[str]
    egress_pii_categories: NotRequired[tuple[str, ...]]
    egress_redaction_count: NotRequired[int]


class IntentDecision(BaseModel):
    intent: QuestionIntent = Field(
        description=(
            "The workflow required to answer the employee's question."
        )
    )
    reason: str = Field(
        description="A short explanation for the selected intent."
    )


intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intent classifier for PolicyPilot, an HR assistant.

Classify the employee question into exactly one intent:

1. policy
   The question asks about a general HR policy, entitlement, rule,
   requirement, deadline, limit, or approval process. Personal employee
   records are not required.

2. employee
   The question asks only about the authenticated employee's stored data,
   such as leave already used during a payroll month.

3. combined
   The question requires both an HR policy and the authenticated employee's
   personal record.

4. out_of_scope
   The question is unrelated to supported HR policy or employee-data
   functionality.

Rules:
- Do not answer the question.
- Do not invent employee information.
- The words "I" and "my" do not automatically mean employee intent.
- Classify according to the information required to answer correctly.

Examples:

Question:
How many sick-leave days do I receive?

Intent:
policy

Reason:
The question asks about the general policy entitlement.

Question:
How many sick-leave days have I used this payroll month?

Intent:
employee

Reason:
The question asks for the authenticated employee's stored usage.

Question:
Can I take five more sick-leave days?

Intent:
combined

Reason:
The answer requires both policy entitlement and personal usage.

Question:
What is the weather today?

Intent:
out_of_scope

Reason:
The question is unrelated to supported HR functionality.
""",
        ),
        (
            "human",
            """
Employee question:
{question}
""",
        ),
    ]
)


intent_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

structured_intent_llm = intent_llm.with_structured_output(
    IntentDecision,
    method="json_schema",
)


def classify_intent(
    state: PolicyState,
) -> dict[str, object]:
    """Classify a sanitized question into a PolicyPilot workflow."""

    formatted_prompt = intent_prompt.invoke(
        {
            "question": state["question"],
        }
    )

    decision = structured_intent_llm.invoke(
        formatted_prompt.messages
    )

    if not isinstance(decision, IntentDecision):
        raise TypeError(
            "The intent classifier returned an invalid result"
        )

    return {
        "intent": decision.intent,
        "intent_reason": decision.reason.strip(),
    }


# This keyword classifier is temporarily retained because the existing
# graph still uses it. It will be removed when we add all four intent routes.
POLICY_KEYWORDS = (
    "leave",
    "payroll",
    "salary",
    "reimbursement",
    "work from home",
    "hr policy",
)


class EvidenceGrade(BaseModel):
    is_applicable: bool = Field(
        description=(
            "True when the context contains policy facts useful for answering "
            "the question, including when the final answer must be conditional. "
            "False only when the context is about a different topic or contains "
            "no useful facts for the question."
        )
    )
    reason: str = Field(
        description="A short explanation of the applicability decision."
    )


grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an HR-policy evidence applicability grader.

Decide whether the retrieved context contains facts useful for answering the
employee's question.

Grading rules:
1. Grade only the question that was actually asked.
2. Do not invent missing employee attributes or additional questions.
3. Mark evidence applicable when it contains a direct answer or supports a
   useful conditional answer.
4. Missing user information does not make evidence inapplicable.
5. A retrieved section is inapplicable when its core topic differs from the
   question. Similar vocabulary alone is insufficient.
6. Different leave categories are not interchangeable.
7. Do not require the policy to cover every possible employee category.
8. Do not answer the question. Only grade the supplied evidence.

Examples:

Question:
How many sick-leave days do I receive?

Context:
Full-time employees receive 10 paid sick-leave days per calendar year.

Grade:
is_applicable = true

Reason:
The context provides the sick-leave entitlement for full-time employees.

Question:
What is the HR policy for parking?

Context:
Full-time employees receive 12 casual-leave days per calendar year.

Grade:
is_applicable = false

Reason:
The context concerns casual leave and contains no parking information.
""",
        ),
        (
            "human",
            """
Employee question:
{question}

Retrieved policy section:
{policy_section}

Retrieved policy context:
{context}
""",
        ),
    ]
)


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are PolicyPilot, a friendly and careful HR policy assistant.

Rules:
1. Answer only from the supplied policy context.
2. Never invent policies, numbers, eligibility rules, or exceptions.
3. Begin with a direct answer.
4. You may make simple comparisons using numbers stated in the context.
5. If the employee's wording is ambiguous, state your interpretation.
6. Mention applicable approval or documentation requirements.
7. Mention the policy section and source naturally.
8. Keep the answer to two to four concise sentences.
9. Recommend contacting HR only for exceptions or information not covered
   by the supplied context.
""",
        ),
        (
            "human",
            """
Employee question:
{question}

Policy section:
{policy_section}

Source:
{source}

Policy context:
{context}
""",
        ),
    ]
)


grader_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

answer_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

structured_grader = grader_model.with_structured_output(
    EvidenceGrade,
    method="json_schema",
)


def grade_evidence(
    state: PolicyState,
) -> dict[str, object]:
    formatted_prompt = grader_prompt.invoke(
        {
            "question": state["question"],
            "policy_section": state["policy_section"],
            "context": state["context"],
        }
    )

    grade = structured_grader.invoke(
        formatted_prompt.messages
    )

    if not isinstance(grade, EvidenceGrade):
        raise TypeError(
            "The evidence grader returned an invalid result"
        )

    return {
        "evidence_applicable": grade.is_applicable,
        "evidence_reason": grade.reason.strip(),
    }


def route_applicability(
    state: PolicyState,
) -> str:
    if state.get("evidence_applicable", False):
        return "applicable"

    return "not_applicable"


def normalize_question(
    state: PolicyState,
) -> dict[str, str]:
    normalized = state["question"].strip().lower()

    return {
        "normalized_question": normalized,
    }


def classify_question(
    state: PolicyState,
) -> dict[str, bool]:
    """Temporary keyword classifier used by the existing graph."""

    normalized_question = state["normalized_question"]

    is_policy_question = any(
        keyword in normalized_question
        for keyword in POLICY_KEYWORDS
    )

    return {
        "is_policy_question": is_policy_question,
    }


def retrieve_policy(
    state: PolicyState,
) -> dict[str, object]:
    matches = vector_store.similarity_search_with_relevance_scores(
        query=state["question"],
        k=1,
    )

    if not matches:
        return {
            "evidence_found": False,
        }

    document, score = matches[0]

    if score < MINIMUM_RELEVANCE:
        return {
            "evidence_found": False,
            "relevance_score": score,
        }

    return {
        "evidence_found": True,
        "policy_section": document.metadata["policy_section"],
        "source": document.metadata["source"],
        "relevance_score": score,
        "context": document.page_content,
    }


def guard_answer(
    answer: str,
) -> dict[str, object]:
    """Apply the egress boundary before an answer enters graph state."""

    result = guard_egress_text(answer)

    return {
        "answer": result.text,
        "egress_action": result.action,
        "egress_pii_categories": result.detected_categories,
        "egress_redaction_count": result.redaction_count,
    }


def reject_question(
    state: PolicyState,
) -> dict[str, object]:
    return guard_answer(
        "PolicyPilot can only answer questions about the available "
        "HR policies."
    )


def route_question(
    state: PolicyState,
) -> str:
    if state["is_policy_question"]:
        return "policy"

    return "not_policy"


def route_evidence(
    state: PolicyState,
) -> str:
    if state.get("evidence_found", False):
        return "evidence"

    return "no_evidence"


def generate_answer(
    state: PolicyState,
) -> dict[str, object]:
    formatted_prompt = answer_prompt.invoke(
        {
            "question": state["question"],
            "policy_section": state["policy_section"],
            "source": state["source"],
            "context": state["context"],
        }
    )

    response = answer_model.invoke(
        formatted_prompt.messages
    )

    return guard_answer(
        response.content.strip()
    )


def no_policy_found(
    state: PolicyState,
) -> dict[str, object]:
    return guard_answer(
        "I could not find relevant information in the available "
        "HR policies. Please contact HR for assistance."
    )


builder = StateGraph(PolicyState)

builder.add_node(
    "normalize_question",
    normalize_question,
)
builder.add_node(
    "classify_question",
    classify_question,
)
builder.add_node(
    "retrieve_policy",
    retrieve_policy,
)
builder.add_node(
    "grade_evidence",
    grade_evidence,
)
builder.add_node(
    "generate_answer",
    generate_answer,
)
builder.add_node(
    "no_policy_found",
    no_policy_found,
)
builder.add_node(
    "reject_question",
    reject_question,
)


builder.add_edge(
    START,
    "normalize_question",
)

builder.add_edge(
    "normalize_question",
    "classify_question",
)


builder.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "policy": "retrieve_policy",
        "not_policy": "reject_question",
    },
)


builder.add_conditional_edges(
    "retrieve_policy",
    route_evidence,
    {
        "evidence": "grade_evidence",
        "no_evidence": "no_policy_found",
    },
)


builder.add_conditional_edges(
    "grade_evidence",
    route_applicability,
    {
        "applicable": "generate_answer",
        "not_applicable": "no_policy_found",
    },
)


builder.add_edge(
    "generate_answer",
    END,
)
builder.add_edge(
    "no_policy_found",
    END,
)
builder.add_edge(
    "reject_question",
    END,
)


_graph = builder.compile()


def run_policy_graph(
    question: str,
) -> PolicyState:
    """Redact sensitive input before it enters LangGraph."""

    ingress_result = redact_ingress_text(question)

    return _graph.invoke(
        {
            "question": ingress_result.text,
            "ingress_pii_categories": (
                ingress_result.detected_categories
            ),
            "ingress_redaction_count": (
                ingress_result.redaction_count
            ),
        }
    )


if __name__ == "__main__":
    questions = [
        "How many sick-leave days do I receive?",
        "How many sick-leave days do full-time employees receive?",
        "What is the HR policy for parking?",
        "What is the weather today?",
    ]

    for question in questions:
        result = run_policy_graph(question)

        print("\nQuestion:", question)
        print(
            "Normalized:",
            result["normalized_question"],
        )
        print(
            "Is policy question:",
            result["is_policy_question"],
        )
        print(
            "Evidence found:",
            result.get("evidence_found"),
        )

        if result.get("evidence_applicable"):
            print(
                "Policy section:",
                result["policy_section"],
            )
            print(
                "Source:",
                result["source"],
            )
            print(
                "Relevance score:",
                round(result["relevance_score"], 3),
            )
            print(
                "Context:",
                result["context"],
            )

        elif result.get("evidence_found"):
            print(
                "Rejected candidate:",
                result["policy_section"],
            )
            print(
                "Candidate score:",
                round(result["relevance_score"], 3),
            )

        if result.get("evidence_reason"):
            print(
                "Evidence applicable:",
                result["evidence_applicable"],
            )
            print(
                "Evidence reason:",
                result["evidence_reason"],
            )

        print(
            "Egress action:",
            result.get("egress_action"),
        )
        print(
            "Answer:",
            result["answer"],
        )