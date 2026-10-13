from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from phase_2_rag.policy_tool_practice import search_hr_policy


model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

SYSTEM_PROMPT = """
You are PolicyPilot, an HR policy assistant.

For every HR policy question:
1. Call search_hr_policy before answering.
2. Use only facts returned by the tool.
3. A retrieved section is potential evidence, not automatic proof.
4. Do not treat different leave categories as interchangeable.
5. If the tool does not provide an applicable policy, say that the topic is
   not covered and recommend contacting HR.
6. Give a concise, friendly answer and mention the source when applicable.
"""

agent = create_agent(
    model=model,
    tools=[search_hr_policy],
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    question = "What is the maternity-leave policy?"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    # Inspect every message in the agent execution.
    for position, message in enumerate(result["messages"], start=1):
        print(f"\n--- Message {position} ---")
        print("Type:", type(message).__name__)
        print("Content:", message.content)

        if getattr(message, "tool_calls", None):
            print("Tool calls:", message.tool_calls)

    # Display only the final response.
    final_message = result["messages"][-1]

    print("\nPolicyPilot:", final_message.content)