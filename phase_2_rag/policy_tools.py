from langchain.tools import tool

from phase_2_rag.vector_store import MINIMUM_RELEVANCE, get_vector_store


vector_store = get_vector_store()


@tool
def search_hr_policy(question: str, max_results: int = 2) -> dict:
    """Search the indexed HR handbook for policy evidence.

    Use this tool before answering questions about employee leave, payroll,
    reimbursements, or work-from-home rules. A returned section is potential
    evidence, not automatic proof that it answers the question. Different
    leave categories must not be treated as interchangeable.

    Args:
        question: The employee's complete HR-policy question.
        max_results: Maximum number of policy sections to return, from 1 to 3.
    """
    question = question.strip()

    if not question:
        return {
            "query": question,
            "matches": [],
            "message": "The policy question cannot be empty.",
        }

    result_limit = max(1, min(max_results, 3))
    scored_results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        # k=result_limit,
        k=5
    )

    matches = [
        {
            "policy_section": document.metadata["policy_section"],
            "source": document.metadata["source"],
            "relevance_score": round(score, 3),
            "content": document.page_content,
        }
        for document, score in scored_results
        if score >= MINIMUM_RELEVANCE
    ]

    return {
        "query": question,
        "minimum_relevance": MINIMUM_RELEVANCE,
        "matches": matches,
        "message": (
            "Review each match for actual applicability before answering."
            if matches
            else "No policy section passed the relevance threshold."
        ),
    }
