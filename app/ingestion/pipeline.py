import os
from functools import lru_cache
from pathlib import Path
from app.ingestion.loader import load_documents
from app.ingestion.splitter import recursive_splitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

from app.ingestion.image_captioner import ImageCaptioner
from app.ingestion.docling_parser import DoclingParser
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.loader import compute_document_id
from app.ingestion.records import IngestionRecord

load_dotenv()


@lru_cache(maxsize=1)
def get_image_captioner() -> ImageCaptioner:
    model_name = os.getenv("OPENROUTER_VISION_MODEL")

    if not model_name:
        raise RuntimeError(
            "OPENROUTER_VISION_MODEL is not configured in .env"
        )

    vision_model = ChatOpenRouter(
        model=model_name,
        temperature=0,
        max_tokens=700,
    )

    return ImageCaptioner(model=vision_model)

@lru_cache(maxsize=1)
def get_docling_parser() -> DoclingParser:
    return DoclingParser()



def ingest_multimodal_pdf(
        file_path: str,
) -> list[IngestionRecord]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    document_id = compute_document_id(path)

    parsed = get_docling_parser().parse_file(
        file_path=str(path),
        document_id=document_id,
    )

    image_extractor = ImageExtractor()

    assets = image_extractor.extract_images(parsed)

    records = list(parsed.records)

    if assets:
        captioner = get_image_captioner()

        image_records = captioner.describe_assets(assets)

        records.extend(image_records)

    return records


def ingest_documents(folder_path: str) -> list[Document]:
    documents = load_documents(folder_path)
    chunks = recursive_splitter(documents)

    if not chunks:
        raise ValueError("No text could be extracted from the provided documents.")

    return chunks
