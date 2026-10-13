from langchain.tools import tool

from phase_2_rag.vector_store import (
    MINIMUM_RELEVANCE,
    get_vector_store,
)


# Load the embedding model and connect to Chroma once.
vector_store = get_vector_store()


@tool
def search_hr_policy(question: str) -> dict:
    """Search the HR policies for information relevant to an employee question."""

    question = question.strip()

    # Guard against an empty question.
    if not question:
        return {
            "found": False,
            "message": "The policy question cannot be empty.",
        }

    # Search Chroma for the closest policy section.
    matches = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=1,
    )

    # Handle an empty Chroma collection.
    if not matches:
        return {
            "found": False,
            "message": "No indexed HR policy was found.",
        }

    # Chroma returns a list of (Document, score) tuples.
    document, score = matches[0]

    # Reject results below the minimum threshold.
    if score < MINIMUM_RELEVANCE:
        return {
            "found": False,
            "relevance_score": round(score, 3),
            "message": (
                "No policy section passed the minimum relevance threshold."
            ),
        }

    # Return the potential policy evidence.
    return {
        "found": True,
        "policy_section": document.metadata["policy_section"],
        "source": document.metadata["source"],
        "relevance_score": round(score, 3),
        "content": document.page_content,
        "message": (
            "Review whether this section actually applies to the question."
        ),
    }


if __name__ == "__main__":
    print("Tool name:", search_hr_policy.name)
    print("Tool description:", search_hr_policy.description)
    print("Tool arguments:", search_hr_policy.args)

    result = search_hr_policy.invoke(
        {
            "question": "What is the maternity-leave policy?",
        }
    )

    print("Tool result:", result)