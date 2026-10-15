
from app.ingestion.pipeline import ingest_documents


def test_ingest_documents():
    chunks = ingest_documents("docs")

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
