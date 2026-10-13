from langchain_text_splitters import MarkdownHeaderTextSplitter

from phase_2_rag.vector_store import DATABASE_PATH, PROJECT_ROOT, get_vector_store


POLICY_PATH = PROJECT_ROOT / "data" / "hr_policy.md"


def ingest_policies() -> None:
    policy_text = POLICY_PATH.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "policy_section")],
        strip_headers=False,
    )
    chunks = header_splitter.split_text(policy_text)

    for chunk in chunks:
        chunk.metadata.update(
            {
                "source": POLICY_PATH.name,
                "document_type": "hr_policy",
            }
        )

    chunk_ids = [
        f"{POLICY_PATH.stem}-{chunk.metadata['policy_section'].lower().replace(' ', '-')}"
        for chunk in chunks
    ]

    vector_store = get_vector_store()
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    print(f"Indexed {len(chunks)} policy sections in {DATABASE_PATH}")
    print("Chunk IDs:")
    for chunk_id in chunk_ids:
        print(f"- {chunk_id}")


if __name__ == "__main__":
    ingest_policies()
