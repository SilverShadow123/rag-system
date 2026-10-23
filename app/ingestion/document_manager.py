import logging
from pathlib import Path

from app.ingestion.loader import compute_document_id, load_pdf
from app.ingestion.splitter import recursive_splitter
from app.retrieval.vector_store import load_vector_store

logger = logging.getLogger(__name__)


def add_document(file_path: str) -> int:
    path = Path(file_path)

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Provide a valid PDF file path.")

    try:
        pages = load_pdf(path)
    except Exception as error:
        raise ValueError(f"Failed to extract text from {path.name}: {error}") from error

    chunks = recursive_splitter(pages)

    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    # load_pdf attaches the document_id to every page it returns.
    document_id = pages[0].metadata["document_id"]

    vector_store = load_vector_store()

    # Make re-ingestion repeatable: drop previous chunks before adding new ones.
    vector_store.delete(where={"document_id": document_id})

    vector_store.add_documents(
        chunks,
        ids=[f"{document_id}-{index:04d}" for index in range(len(chunks))],
    )

    return len(chunks)


def delete_document(file_path: str) -> None:
    path = Path(file_path)
    vector_store = load_vector_store()

    if path.is_file():
        vector_store.delete(where={"document_id": compute_document_id(path)})

    vector_store.delete(where={"source": path.name})

    if path.is_file():
        path.unlink()