
import logging
import shutil
from pathlib import Path

from app.ingestion.loader import compute_document_id
from app.ingestion.pipeline import ingest_and_index_multimodal_pdf
from app.retrieval.vector_store import load_vector_store


logger = logging.getLogger(__name__)

IMAGE_ROOT = Path("data/images")


def add_document(file_path: str) -> int:
    """
    Ingest a PDF through the multimodal pipeline and index its records.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    try:
        indexed_count = ingest_and_index_multimodal_pdf(
            str(path)
        )

    except Exception:
        logger.exception(
            "Multimodal ingestion failed: %s",
            path.name,
        )
        raise

    logger.info(
        "Successfully indexed %d records from %s",
        indexed_count,
        path.name,
    )

    return indexed_count


def delete_document(file_path: str) -> None:
    """
    Delete a PDF's indexed records, extracted images, and original file.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    # Calculate the content-based document ID before deleting the PDF.
    document_id = compute_document_id(path)

    # Remove all ChromaDB records belonging to this PDF.
    vector_store = load_vector_store()

    vector_store.delete(
        where={
            "document_id": document_id,
        }
    )

    # Remove extracted image assets belonging to this PDF.
    document_image_dir = IMAGE_ROOT / document_id

    if document_image_dir.exists():
        shutil.rmtree(document_image_dir)

    # Remove the original PDF.
    path.unlink()

    logger.info(
        "Deleted document and indexed assets: %s",
        path.name,
    )
