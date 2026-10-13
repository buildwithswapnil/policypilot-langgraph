from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from phase_2_rag.policy_tools import search_hr_policy



SYSTEM_PROMPT = """
You are PolicyPilot, a friendly and careful HR policy assistant.

For every HR policy question:
1. Call search_hr_policy before answering.
2. Use only facts returned by the tool. Never use outside knowledge.
3. Treat retrieved sections as potential evidence. Verify that the section's
   policy category actually matches the employee's question.
4. Never treat sick, casual, maternity, or parental leave as interchangeable.
5. You may perform simple comparisons using numbers explicitly stated by the
   policy evidence.
6. Start with a direct answer and keep the response to two to four sentences.
7. Mention the applicable policy section and source when evidence supports the
   answer.
8. If no returned section actually answers the question, say that the topic is
   not covered in the available HR policies and recommend contacting HR.
"""

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

policy_agent = create_agent(
    model=model,
    tools=[search_hr_policy],
    system_prompt=SYSTEM_PROMPT,
)


def run_policy_agent(question: str) -> str:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    result = policy_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return result["messages"][-1].content


def main() -> None:
    question = input("\nAsk the PolicyPilot agent a question: ")

    try:
        answer = run_policy_agent(question)
    except ValueError as error:
        print(f"\nPolicyPilot: {error}")
        return

    print(f"\nPolicyPilot agent: {answer}")


# if __name__ == "__main__":
#     main()
main() 