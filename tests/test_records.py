import pytest
from pydantic import ValidationError

from app.ingestion.records import IngestionRecord, make_record_id


def make_valid_record(**overrides):
    data = {
        "record_id": "doc123-p1-text-e0-c0",
        "document_id": "doc123",
        "source": "sample.pdf",
        "page_number": 1,
        "element_type": "text",
        "element_index": 0,
        "chunk_index": 0,
        "content": "This is sample content.",
        "extractor": "docling",
    }

    data.update(overrides)

    return data
def test_valid_text_record():
    record = IngestionRecord(**make_valid_record())

    assert record.element_type == "text"
def test_valid_table_record():
    record = IngestionRecord(
        **make_valid_record(element_type="table")
    )

    assert record.element_type == "table"
def test_valid_image_description():
    record = IngestionRecord(
        **make_valid_record(
            element_type="image_description",
            image_path="data/images/doc123/page_001_image_001.png",
        )
    )

    assert record.image_path is not None
def test_image_description_requires_image_path():
    with pytest.raises(ValidationError):
        IngestionRecord(
            **make_valid_record(element_type="image_description")
        )

def test_empty_content_rejected():
    with pytest.raises(ValidationError):
        IngestionRecord(**make_valid_record(content="   "))
def test_invalid_page_number():
    with pytest.raises(ValidationError):
        IngestionRecord(**make_valid_record(page_number=0))

def test_negative_element_index():
    with pytest.raises(ValidationError):
        IngestionRecord(**make_valid_record(element_index=-1))

def test_unexpected_field_rejected():
    with pytest.raises(ValidationError):
        IngestionRecord(
            **make_valid_record(unexpected_field="not allowed")
        )

def test_record_id_is_deterministic():
    first = make_record_id("doc123", 1, "text", 0, 0)
    second = make_record_id("doc123", 1, "text", 0, 0)

    assert first == second

