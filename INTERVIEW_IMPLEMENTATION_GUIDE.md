# PolicyPilot AI — Interview Implementation Snippets

> Purpose: concise, copy-paste-friendly notes for explaining the PolicyPilot proof of concept.
>
> Status: ingestion, embeddings, Chroma retrieval, LangChain generation, basic LangGraph routing, guardrails, local evaluations, and FastAPI health endpoints exist. Langfuse and the FastAPI chat endpoint below are the recommended next implementation, not completed features.

## 1. End-to-end architecture

```text
HR policy Markdown
    -> header-aware chunking
    -> Hugging Face embeddings
    -> Chroma vector database

Employee question
    -> ingress PII redaction
    -> LangGraph routing
    -> Chroma similarity retrieval
    -> evidence relevance check
    -> LangChain prompt + Ollama Llama 3.2
    -> egress PII guardrail
    -> FastAPI response
    -> sanitized Langfuse trace (planned)
```

Interview summary:

> I built a local, privacy-aware HR-policy RAG system. Policies are split by Markdown section, embedded with MiniLM, and persisted in Chroma. LangGraph controls routing and evidence checks, while LangChain constructs prompts and invokes Llama 3.2 through Ollama. Ingress and egress guardrails prevent PII leakage, and retrieval and answer evaluations detect regressions.

## 2. Document ingestion and chunking

Actual approach: split `hr_policy.md` at each level-two Markdown heading so every chunk represents one policy section.

```python
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

POLICY_PATH = Path("data/hr_policy.md")

text = POLICY_PATH.read_text(encoding="utf-8")

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "policy_section")],
    strip_headers=False,
)
chunks = splitter.split_text(text)

for chunk in chunks:
    chunk.metadata.update({
        "source": POLICY_PATH.name,
        "document_type": "hr_policy",
    })
```

Why:

- Semantic policy boundaries are better than arbitrary character boundaries.
- Metadata supports citations, filtering, and debugging.
- Stable sections reduce the chance of mixing unrelated leave policies.

## 3. Convert text into embeddings

Actual embedding model: `sentence-transformers/all-MiniLM-L6-v2`.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Chroma invokes this internally for each document and query.
vector = embeddings.embed_query("How many sick-leave days do I receive?")
```

Interview explanation:

> An embedding converts text into a numerical vector. Semantically similar policy text and questions should be close in vector space. Normalization makes cosine-based comparison consistent.

## 4. Store embeddings in Chroma

Actual vector database: persistent local Chroma using cosine distance.

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="policypilot_hr_policies_v3",
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_configuration={"hnsw": {"space": "cosine"}},
)

chunk_ids = [
    f"hr-policy-{chunk.metadata['policy_section'].lower().replace(' ', '-')}"
    for chunk in chunks
]

vector_store.add_documents(documents=chunks, ids=chunk_ids)
```

Why stable IDs:

> Stable IDs make re-ingestion predictable and help avoid uncontrolled duplicate records. After changing the source policy, the ingestion job must run again so Chroma contains the new embeddings and text.

## 5. Retrieve relevant policy evidence

```python
MINIMUM_RELEVANCE = 0.30

matches = vector_store.similarity_search_with_relevance_scores(
    query=question,
    k=5,
)

relevant = [
    (document, score)
    for document, score in matches
    if score >= MINIMUM_RELEVANCE
]

if not relevant:
    return "I could not find relevant information in the available HR policies."

context = "\n\n".join(
    f"Section: {doc.metadata['policy_section']}\n"
    f"Source: {doc.metadata['source']}\n{doc.page_content}"
    for doc, _ in relevant
)
```

Important interview point:

> Vector similarity is candidate retrieval, not proof that the evidence answers the question. I use a threshold and a separate applicability check to reduce false-positive answers, such as returning casual-leave policy for a parking question.

## 6. LangChain answer generation

LangChain handles prompt composition, model invocation, and structured output.

```python
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class GeneratedAnswer(BaseModel):
    answer: str
    support_status: Literal["supported", "partial", "unsupported"] = Field(
        description="How strongly the supplied context supports the answer"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are PolicyPilot, an HR policy assistant.
Answer only from the supplied context.
Do not invent policies, numbers, or exceptions.
If evidence is partial, state the known rule and recommend contacting HR.
Keep the answer concise and mention the policy section.
"""),
    ("human", "Policy context:\n{context}\n\nQuestion:\n{question}"),
])

llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
structured_llm = llm.with_structured_output(
    GeneratedAnswer,
    method="json_schema",
)

messages = prompt.invoke({"context": context, "question": question})
result = structured_llm.invoke(messages.messages)
```

Interview explanation:

> LangChain is the component-integration layer. It standardizes prompts, documents, model calls, tools, and structured outputs. It does not decide the complete workflow; LangGraph performs that orchestration.

## 7. LangGraph workflow orchestration

Current implemented policy path:

```text
START -> normalize -> classify -> retrieve -> grade evidence
                                           -> generate -> END
                                           -> fallback -> END
                         out of scope -> reject -> END
```

Condensed implementation:

```python
from typing import NotRequired, TypedDict #TypedDict  → structure of shared graph state
from langgraph.graph import START, END, StateGraph

class PolicyState(TypedDict):
    question: str
    is_policy_question: NotRequired[bool]
    evidence_found: NotRequired[bool]
    evidence_applicable: NotRequired[bool]
    context: NotRequired[str]
    answer: NotRequired[str]

builder = StateGraph(PolicyState)

builder.add_node("classify", classify_question)
builder.add_node("retrieve", retrieve_policy)
builder.add_node("grade", grade_evidence)
builder.add_node("generate", generate_answer)
builder.add_node("fallback", no_policy_found)
builder.add_node("reject", reject_question)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_question,
    {"policy": "retrieve", "not_policy": "reject"},
)
builder.add_conditional_edges(
    "retrieve",
    route_evidence,
    {"evidence": "grade", "no_evidence": "fallback"},
)
builder.add_conditional_edges(
    "grade",
    route_applicability,
    {"applicable": "generate", "not_applicable": "fallback"},
)
builder.add_edge("generate", END)
builder.add_edge("fallback", END)
builder.add_edge("reject", END)

policy_graph = builder.compile()
```

Interview explanation:

> LangGraph gives me explicit state, nodes, conditional edges, and deterministic fallback paths. This is safer and easier to test than hiding all workflow decisions inside one agent prompt.

Planned extension: replace the temporary binary classifier with four routes: `policy`, `employee`, `combined`, and `out_of_scope`.

## 8. Ingress guardrail

Redact sensitive text before retrieval, model calls, or tracing.

```python
import re

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I),
    "aadhaar": re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
    "secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}

def redact_ingress(text: str) -> str:
    if not text.strip():
        raise ValueError("Question cannot be empty")

    safe_text = text.strip()
    for category, pattern in PII_PATTERNS.items():
        safe_text = pattern.sub(f"[{category.upper()}_REDACTED]", safe_text)
    return safe_text
```

## 9. Structured employee-data guardrail

Drop direct identifiers, tokenize the employee code, allow safe fields, and withhold restricted fields.

```python
import hashlib
import hmac

COLUMN_RULES = {
    "First Name": "drop",
    "Last Name": "drop",
    "Email": "drop",
    "Employee Code": "tokenize",
    "Department": "allow",
    "Designation": "allow",
    "Monthly Salary": "restrict",
}

def tokenize_identifier(value: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode(),
        value.strip().upper().encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"employee_{digest[:16]}"

def sanitize_row(row: dict[str, str], secret: str) -> dict[str, str]:
    safe = {}
    for column, value in row.items():
        action = COLUMN_RULES[column]
        if action == "allow":
            safe[column] = value
        elif action == "tokenize":
            safe["Employee Token"] = tokenize_identifier(value, secret)
    return safe
```

Why HMAC instead of a plain hash:

> HMAC uses a server-side secret, making low-entropy employee identifiers harder to reverse through dictionary attacks while still producing stable lookup tokens.

## 10. Egress guardrail

Inspect the generated answer before returning it.

```python
HIGH_RISK = {"pan", "aadhaar", "bank_account", "secret"}

def guard_egress(answer: str) -> str:
    if not answer.strip():
        return "I’m unable to return that response safely."

    result = detect_and_redact_pii(answer)
    detected = set(result.detected_categories)

    if detected & HIGH_RISK:
        return "I’m unable to return that response because it may contain sensitive information."
    if result.redaction_count:
        return result.text             # Redact lower-risk email/phone data.
    return answer
```

Defense-in-depth summary:

```text
raw input -> ingress redaction -> authorized context -> LLM
          -> egress inspection -> sanitized trace -> user
```

## 11. Retrieval evaluation

Use a small, representative golden dataset—not one test per source document.

```python
TEST_CASES = [
    ("How many casual-leave days do I receive?", "Casual Leave Policy"),
    ("Do I need a certificate after three sick days?", "Sick Leave Policy"),
    ("What is the parking policy?", None),  # Must reject.
]

passed = 0
for question, expected_section in TEST_CASES:
    matches = vector_store.similarity_search_with_relevance_scores(question, k=1)
    document, score = matches[0]
    predicted = document.metadata["policy_section"] if score >= 0.30 else None
    passed += predicted == expected_section

accuracy = passed / len(TEST_CASES)
print(f"Retrieval accuracy: {accuracy:.1%}")
```

For a larger system, report `Recall@K`, `Precision@K`, mean reciprocal rank, and rejection accuracy by policy category.

## 12. Answer evaluation

```python
TEST_CASES = [{
    "question": "Can I take more than 6 sick leaves?",
    "required": [("10", "ten"), ("medical certificate",)],
    "forbidden": ("unlimited", "not covered"),
}]

def answer_passes(answer: str, case: dict) -> bool:
    normalized = answer.casefold().replace("-", " ")
    has_required = all(
        any(option in normalized for option in alternatives)
        for alternatives in case["required"]
    )
    has_forbidden = any(term in normalized for term in case["forbidden"])
    return has_required and not has_forbidden
```

Interview explanation:

> Retrieval evaluation asks whether the correct evidence was found. Answer evaluation asks whether the final response contains required facts and avoids forbidden claims. Keeping these separate tells me whether a failure came from retrieval or generation.

## 13. Langfuse observability — planned implementation

This is a safe template, not yet implemented in PolicyPilot. Trace only sanitized values and disable automatic root input/output capture.
# CallbackHandler automatically traces LangChain operations
# @observe becomes important when you want one parent trace covering the complete custom workflow—guardrails, retrieval, business logic, FastAPI processing, and the model call—not just LangChain operations.

```python
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

@observe(
    name="policy-pilot-request",
    as_type="agent",
    capture_input=False,
    capture_output=False,
)
def run_traced_policy_graph(raw_question: str, request_id: str):
    ingress = redact_ingress_text(raw_question)

    with propagate_attributes(
        session_id=request_id,
        tags=["policy-pilot", "rag"],
        metadata={"pii_redactions": str(ingress.redaction_count)},
    ):
        result = policy_graph.invoke(
            {"question": ingress.text},
            config={"callbacks": [langfuse_handler]},
        )

    return guard_egress_text(result["answer"])
```

Required environment variables:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

Production rule:

> Redact before tracing and add a Langfuse export-time masking function as a second privacy layer. Never attach raw employee records, authentication tokens, or secrets to trace metadata.

## 14. FastAPI dependency injection and health API

Actual settings dependency:

```python
from functools import lru_cache
from typing import Annotated
from fastapi import Depends
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "PolicyPilot API"
    api_v1_prefix: str = "/api/v1"

@lru_cache
def get_settings() -> Settings:
    return Settings()

SettingsDependency = Annotated[Settings, Depends(get_settings)]
```

Actual health endpoint pattern:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/live")
def liveness(settings: SettingsDependency):
    return {"status": "ok", "service": settings.app_name}
```

## 15. FastAPI chat endpoint — recommended next implementation

Separate HTTP handling from business logic through an injected service.

```python
from typing import Annotated
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    answer: str
    policy_section: str | None = None
    support_status: str

class PolicyService:
    def ask(self, question: str) -> ChatResponse:
        result = run_policy_graph(question)
        return ChatResponse(
            answer=result["answer"],
            policy_section=result.get("policy_section"),
            support_status=("supported" if result.get("evidence_applicable") else "unsupported"),
        )

def get_policy_service(request: Request) -> PolicyService:
    return request.app.state.policy_service

PolicyServiceDependency = Annotated[
    PolicyService,
    Depends(get_policy_service),
]

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def ask_policy(body: ChatRequest, service: PolicyServiceDependency):
    return service.ask(body.question)
```

Initialize the shared service once during application lifespan:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.policy_service = PolicyService()
    app.state.ready = True
    yield
    app.state.ready = False

app = FastAPI(lifespan=lifespan)
```

Request flow:

```text
POST /api/v1/chat
  -> Pydantic validates input
  -> FastAPI injects PolicyService
  -> service invokes protected LangGraph workflow
  -> Pydantic validates output
  -> JSON response
```

## 16. Thirty-second interview answer

> PolicyPilot is a local HR-policy RAG application. I ingest Markdown policies using header-aware chunks, generate normalized MiniLM embeddings, and persist them in a cosine-based Chroma collection. For each question, ingress guardrails redact PII, LangGraph routes and retrieves evidence, an LLM-based grader checks applicability, and LangChain invokes Llama 3.2 through Ollama with structured output. An egress guardrail blocks or redacts sensitive output. I separately evaluate retrieval and generated answers, expose the workflow through FastAPI with dependency injection, and plan to send sanitized traces and latency metadata to Langfuse.

## 17. Honest project-status language

Say:

- “Implemented” for ingestion, embeddings, Chroma, LangChain RAG, basic policy LangGraph, guardrails, evaluations, and health endpoints.
- “Partially implemented” for four-way LangGraph routing.
- “Designed/planned” for Langfuse and the FastAPI chat endpoint until their code and tests are complete.

Do not say that Langfuse, full employee/combined routing, Deep Agents, or production deployment are complete yet.

