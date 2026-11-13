from app.ingestion.records import IngestionRecord
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks: list[Document]) -> Chroma:
    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function=create_embeddings(),
        persist_directory="data/chroma",
    )

    if chunks:
        vector_store.add_documents(chunks)

    return vector_store


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name="rag_documents",
        embedding_function=create_embeddings(),
        persist_directory="data/chroma",
    )


def create_retriever(vectorstore: Chroma, k: int = 4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

def add_ingestion_records(
    records: list[IngestionRecord],
) -> int:

    if not records:
        raise ValueError("No ingestion records provided.")

    ids = [
        record.record_id
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record IDs detected.")

    documents = [
        record.to_langchain_document()
        for record in records
    ]

    vector_store = load_vector_store()

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return len(documents)
