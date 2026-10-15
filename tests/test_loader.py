
from app.ingestion.loader import load_documents
from app.ingestion.splitter import recursive_splitter


def test_load_and_split_documents():
    documents = load_documents("docs")
    chunks = recursive_splitter(documents)

    assert len(documents) > 0
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
