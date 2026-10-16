
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf"}
HASH_BUFFER_SIZE = 1024 * 1024


def compute_document_id(file_path: str | Path) -> str:
    """Return a stable SHA-256 identifier based on file contents."""
    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(HASH_BUFFER_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and attach normalized source metadata."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    document_id = compute_document_id(path)

    pages = PyPDFLoader(
        str(path),
        extraction_mode="plain",
    ).load()

    documents: list[Document] = []

    for page in pages:
        content = page.page_content.strip()

        if not content:
            logger.debug(
                "Skipping empty page %s of %s",
                page.metadata.get("page"),
                path.name,
            )
            continue

        metadata = {
            **page.metadata,
            "source": path.name,
            "document_id": document_id,
        }

        page_number = metadata.get("page")
        if isinstance(page_number, int):
            metadata["page"] = page_number + 1

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    if not documents:
        logger.warning(
            "No extractable text found in PDF: %s. "
            "The PDF may be scanned, empty, or image-based.",
            path.name,
        )

    logger.info(
        "Loaded %d pages from %s",
        len(documents),
        path.name,
    )

    return documents


def load_documents(folder_path: str | Path) -> list[Document]:
    """Load all supported PDFs from a directory."""
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Document folder not found: {folder}"
        )

    documents: list[Document] = []
    failed_files: list[str] = []

    pdf_files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )

    if not pdf_files:
        logger.warning("No PDF files found in %s", folder)

    for file_path in pdf_files:
        try:
            documents.extend(load_pdf(file_path))
        except Exception:
            failed_files.append(file_path.name)
            logger.exception(
                "Failed to load PDF: %s",
                file_path.name,
            )

    logger.info(
        "Ingestion summary: %d PDFs found, %d pages loaded, "
        "%d PDFs failed",
        len(pdf_files),
        len(documents),
        len(failed_files),
    )

    if failed_files:
        logger.warning(
            "Failed PDF files: %s",
            ", ".join(failed_files),
        )

    return documents
