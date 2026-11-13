from app.ingestion.pipeline import (
    ingest_and_index_multimodal_pdf,
)

indexed_count = ingest_and_index_multimodal_pdf(
    "docs/2401.01473v3.pdf"
)

print("Indexed records:", indexed_count)