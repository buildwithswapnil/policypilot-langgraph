from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are PolicyPilot, a helpful HR policy assistant.",
        ),
        (
            "human",
            "Employee question: {question}",
        ),
    ]
)

model = FakeListChatModel(
    responses=[
        "Employees receive 12 days of casual leave per calendar year."
    ]
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = input("Ask PolicyPilot: ")
formatted_prompt = prompt.invoke(
    {
        "question": question,
    }
)

print("\n--- Formatted messages ---")

for message in formatted_prompt.messages:
    print(f"{message.type.upper()}: {message.content}")

print("--------------------------")

answer = chain.invoke(
    {
        "question": question,
    }
)

print("\nPolicyPilot:", answer)
