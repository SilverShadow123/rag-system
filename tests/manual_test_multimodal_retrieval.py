
from pathlib import Path

from app.ingestion.pipeline import ingest_and_index_multimodal_pdf
from app.ingestion.loader import compute_document_id
from app.retrieval.vector_store import load_vector_store


# Configuration
PDF_PATH = Path("docs/2401.01473v3.pdf")
QUERY = "What are the main findings of this research?"
TOP_K = 4


def main() -> None:

    # 1. Validate the PDF
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    print("=" * 70)
    print("MULTIMODAL RETRIEVAL TEST")
    print("=" * 70)

    # 2. Ingest and index the PDF
    print("\n[1] Ingesting and indexing PDF...")

    indexed_count = ingest_and_index_multimodal_pdf(
        str(PDF_PATH)
    )

    print(f"Indexed records: {indexed_count}")

    # 3. Calculate the document ID
    document_id = compute_document_id(PDF_PATH)

    print(f"Document ID: {document_id}")

    # 4. Load ChromaDB
    print("\n[2] Loading ChromaDB...")

    vector_store = load_vector_store()

    print("Vector store loaded successfully.")

    # 5. Perform document-filtered retrieval
    print("\n[3] Performing retrieval...")

    print(f"Query: {QUERY}")
    print(f"Top K: {TOP_K}")

    results = vector_store.similarity_search(
        query=QUERY,
        k=TOP_K,
        filter={
            "document_id": document_id,
        },
    )

    # 6. Display retrieval results
    print("\n[4] RETRIEVED RECORDS")
    print("=" * 70)

    if not results:
        print("No matching records found.")
        return

    for index, doc in enumerate(results, start=1):

        metadata = doc.metadata

        print(f"\n--- Result {index} ---")

        print(
            "Source:",
            metadata.get("source", "Unknown"),
        )

        print(
            "Document ID:",
            metadata.get("document_id", "Unknown"),
        )

        print(
            "Page:",
            metadata.get(
                "page_number",
                metadata.get("page", "Unknown"),
            ),
        )

        print(
            "Element Type:",
            metadata.get("element_type", "Unknown"),
        )

        print(
            "Element Index:",
            metadata.get("element_index", "Unknown"),
        )

        print(
            "Chunk Index:",
            metadata.get("chunk_index", "Unknown"),
        )

        print(
            "Image Path:",
            metadata.get("image_path", "None"),
        )

        print("\nContent:")
        print(doc.page_content[:700])

        print("-" * 70)

    # 7. Validate document filtering
    retrieved_document_ids = {
        doc.metadata.get("document_id")
        for doc in results
    }

    print("\n[5] VALIDATION")
    print("=" * 70)

    print(f"Results retrieved: {len(results)}")
    print(f"Expected document ID: {document_id}")
    print(f"Retrieved document IDs: {retrieved_document_ids}")

    if retrieved_document_ids == {document_id}:
        print("SUCCESS: All results belong to the target PDF.")
    else:
        print("FAILURE: Unexpected document IDs found.")

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    main()
