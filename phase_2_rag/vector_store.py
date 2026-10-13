from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "policypilot_hr_policies_v3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINIMUM_RELEVANCE = 0.30


def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DATABASE_PATH),
        collection_configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    return vector_store
