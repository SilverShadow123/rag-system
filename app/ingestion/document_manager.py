
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader

from app.ingestion.splitter import recursive_splitter
from app.retrieval.vector_store import load_vector_store


def add_document(file_path: str) -> int:
    path = Path(file_path)

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Provide a valid PDF file path.")

    pages = PyPDFLoader(str(path), extraction_mode="plain").load()

    for page in pages:
        page.metadata["source"] = str(path)

    chunks = recursive_splitter(pages)

    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    vector_store = load_vector_store()

    vector_store.add_documents(
        chunks,
        ids=[str(uuid4()) for _ in chunks],
    )

    return len(chunks)


def delete_document(file_path: str) -> None:
    path = Path(file_path)
    vector_store = load_vector_store()

    vector_store.delete(where={"source": str(path)})

    if path.is_file():
        path.unlink()
