
from unittest.mock import MagicMock

import pytest

from app.ingestion.records import IngestionRecord
from app.retrieval import vector_store


def make_record(**overrides):

    data = {
        "record_id": "doc123-p1-text-e0-c0",
        "document_id": "doc123",
        "source": "sample.pdf",
        "page_number": 1,
        "element_type": "text",
        "element_index": 0,
        "chunk_index": 0,
        "content": "Sample document content.",
        "extractor": "docling",
    }

    data.update(overrides)

    return IngestionRecord.model_validate(data)


@pytest.fixture
def mock_store(monkeypatch):

    store = MagicMock()

    monkeypatch.setattr(
        vector_store,
        "load_vector_store",
        lambda: store,
    )

    return store


def test_add_ingestion_records(mock_store):

    records = [
        make_record(),
        make_record(
            record_id="doc123-p1-text-e1-c0",
            element_index=1,
            content="Second paragraph.",
        ),
    ]

    count = vector_store.add_ingestion_records(records)

    assert count == 2

    mock_store.add_documents.assert_called_once()

    call = mock_store.add_documents.call_args

    assert call.kwargs["ids"] == [
        "doc123-p1-text-e0-c0",
        "doc123-p1-text-e1-c0",
    ]

    assert len(call.kwargs["documents"]) == 2


def test_records_are_converted_to_documents(mock_store):

    record = make_record()

    vector_store.add_ingestion_records([record])

    call = mock_store.add_documents.call_args

    document = call.kwargs["documents"][0]

    assert document.page_content == record.content
    assert document.metadata["document_id"] == record.document_id
    assert document.metadata["page_number"] == record.page_number


def test_empty_records_are_rejected(mock_store):

    with pytest.raises(ValueError):
        vector_store.add_ingestion_records([])

    mock_store.add_documents.assert_not_called()


def test_duplicate_ids_are_rejected(mock_store):

    records = [
        make_record(),
        make_record(),
    ]

    with pytest.raises(ValueError):
        vector_store.add_ingestion_records(records)

    mock_store.add_documents.assert_not_called()


def test_image_path_is_preserved(mock_store):

    record = make_record(
        element_type="image_description",
        image_path="data/images/doc123/page_0001_picture_000.png",
    )

    vector_store.add_ingestion_records([record])

    document = (
        mock_store.add_documents
        .call_args.kwargs["documents"][0]
    )

    assert (
        document.metadata["image_path"]
        == record.image_path
    )
