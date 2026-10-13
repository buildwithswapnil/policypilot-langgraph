from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

response = llm.invoke(
    "Reply with exactly: LangChain and Ollama are connected."
)

print("Response type:", type(response).__name__)
print("Response content:", response.content)