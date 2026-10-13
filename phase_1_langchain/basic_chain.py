import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

llm = ChatOllama(
    model=model_name,
    temperature=0,
)


if not model_name:
    raise ValueError("OPENAI_MODEL was not found")

llm = ChatOllama(
    model=model_name,
    use_responses_api=True,
)

response = llm.invoke(
    "Reply with exactly this sentence: PolicyPilot connection successful."
)

print("Response type:", type(response).__name__)
print("Model response:", response.content)