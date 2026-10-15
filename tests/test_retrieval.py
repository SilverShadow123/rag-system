from app.ingestion.pipeline import ingest_documents
from app.retrieval.vector_store import create_vector_store, create_retriever


def test_retrieval():
    chunks = ingest_documents("docs")

    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store, k=4)

    query = "An AISE officer operating under commercial cover in Cyprus received a secure tasking"
    results = retriever.invoke(query)

    print(f"Total chunks: {len(chunks)}")
    print(f"Retrieved chunks: {len(results)}")

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content)
        print("Metadata:", document.metadata)


if __name__ == "__main__":
    test_retrieval()