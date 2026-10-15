
from app.ingestion.pipeline import ingest_documents
from app.retrieval.vector_store import create_vector_store, create_retriever
from app.generation.generator import generate_answer


def test_rag_flow():
    chunks = ingest_documents("docs")
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store, k=4)

    question = "What is a rag system?"
    documents = retriever.invoke(question)

    answer = generate_answer(question, documents)

    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print("\nRetrieved sources:")

    for document in documents:
        print(
            document.metadata.get("source"),
            "Page:",
            document.metadata.get("page"),
        )

    assert isinstance(answer, str)
    assert answer.strip()


if __name__ == "__main__":
    test_rag_flow()
