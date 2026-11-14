
from unittest.mock import MagicMock, call

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

    # Existing records must be deleted first.
    mock_store.delete.assert_called_once_with(
        where={"document_id": "doc123"}
    )

    # New records must then be inserted.
    mock_store.add_documents.assert_called_once()

    call_args = mock_store.add_documents.call_args

    assert call_args.kwargs["ids"] == [
        "doc123-p1-text-e0-c0",
        "doc123-p1-text-e1-c0",
    ]

    assert len(call_args.kwargs["documents"]) == 2


def test_existing_records_are_deleted_before_insertion(mock_store):

    record = make_record()

    vector_store.add_ingestion_records([record])

    expected_calls = [
        call.delete(where={"document_id": "doc123"}),
        call.add_documents(
            documents=[record.to_langchain_document()],
            ids=["doc123-p1-text-e0-c0"],
        ),
    ]

    assert mock_store.mock_calls == expected_calls


def test_multiple_document_ids_are_deleted(mock_store):

    records = [
        make_record(
            record_id="doc123-p1-text-e0-c0",
            document_id="doc123",
        ),
        make_record(
            record_id="doc456-p1-text-e0-c0",
            document_id="doc456",
            source="another.pdf",
        ),
    ]

    count = vector_store.add_ingestion_records(records)

    assert count == 2

    # Both document IDs must be deleted.
    deleted_document_ids = {
        item.kwargs["where"]["document_id"]
        for item in mock_store.delete.call_args_list
    }

    assert deleted_document_ids == {
        "doc123",
        "doc456",
    }

    assert mock_store.delete.call_count == 2


def test_records_are_converted_to_documents(mock_store):

    record = make_record()

    vector_store.add_ingestion_records([record])

    documents = (
        mock_store.add_documents.call_args.kwargs["documents"]
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == record.content
    assert document.metadata["record_id"] == record.record_id
    assert document.metadata["document_id"] == record.document_id
    assert document.metadata["source"] == record.source
    assert document.metadata["page_number"] == record.page_number
    assert document.metadata["element_type"] == record.element_type
    assert document.metadata["element_index"] == record.element_index
    assert document.metadata["chunk_index"] == record.chunk_index


def test_empty_records_are_rejected(mock_store):

    with pytest.raises(ValueError, match="No ingestion records provided"):
        vector_store.add_ingestion_records([])

    mock_store.delete.assert_not_called()
    mock_store.add_documents.assert_not_called()


def test_duplicate_ids_are_rejected(mock_store):

    records = [
        make_record(),
        make_record(),
    ]

    with pytest.raises(ValueError, match="Duplicate record IDs"):
        vector_store.add_ingestion_records(records)

    mock_store.delete.assert_not_called()
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

    assert document.metadata["image_path"] == record.image_path


def test_table_records_are_inserted(mock_store):

    record = make_record(
        record_id="doc123-p1-table-e0-c0",
        element_type="table",
        content=(
            "| Model | Accuracy |\n"
            "|---|---|\n"
            "| Model A | 95% |"
        ),
    )

    count = vector_store.add_ingestion_records([record])

    assert count == 1

    document = (
        mock_store.add_documents
        .call_args.kwargs["documents"][0]
    )

    assert document.page_content == record.content
    assert document.metadata["element_type"] == "table"


def test_insertion_uses_deterministic_record_ids(mock_store):

    records = [
        make_record(),
        make_record(
            record_id="doc123-p1-text-e1-c0",
            element_index=1,
            content="Another paragraph.",
        ),
    ]

    vector_store.add_ingestion_records(records)

    ids = mock_store.add_documents.call_args.kwargs["ids"]

    assert ids == [
        record.record_id
        for record in records
    ]
