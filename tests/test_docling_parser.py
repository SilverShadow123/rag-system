
from pathlib import Path

import pytest
from docling_core.types.doc import PictureItem

from app.ingestion.docling_parser import (
    DoclingParser,
    ParsedPDF,
)
from app.ingestion.records import IngestionRecord


# Change this to the path of your test PDF.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = PROJECT_ROOT / "docs" / "2401.01473v3.pdf"


@pytest.fixture(scope="module")
def parser():
    return DoclingParser()


@pytest.fixture(scope="module")
def parsed_pdf(parser):
    if not SAMPLE_PDF.is_file():
        pytest.fail(
            f"Test PDF not found: {SAMPLE_PDF}. "
            "Update SAMPLE_PDF in test_docling_parser.py."
        )

    return parser.parse_file(
        file_path=str(SAMPLE_PDF),
        document_id="test_document_001",
    )


def test_parser_returns_parsed_pdf(parsed_pdf):
    assert isinstance(parsed_pdf, ParsedPDF)
    assert parsed_pdf.document_id == "test_document_001"
    assert parsed_pdf.source == SAMPLE_PDF.name
    assert parsed_pdf.document is not None
    assert isinstance(parsed_pdf.markdown, str)
    assert isinstance(parsed_pdf.status, str)


def test_records_are_valid(parsed_pdf):
    assert all(
        isinstance(record, IngestionRecord)
        for record in parsed_pdf.records
    )

    for record in parsed_pdf.records:
        assert record.content.strip()
        assert record.page_number >= 1
        assert record.element_type in ("text", "table")
        assert record.document_id == "test_document_001"


def test_text_extraction(parsed_pdf):
    text_records = [
        record
        for record in parsed_pdf.records
        if record.element_type == "text"
    ]

    assert len(text_records) > 0, "No text records extracted."


def test_table_extraction(parsed_pdf):
    table_records = [
        record
        for record in parsed_pdf.records
        if record.element_type == "table"
    ]

    assert len(table_records) > 0, "No table records extracted."

    for record in table_records:
        assert record.content.strip()


def test_record_ids_are_unique(parsed_pdf):
    record_ids = [
        record.record_id
        for record in parsed_pdf.records
    ]

    assert len(record_ids) == len(set(record_ids))


def test_picture_candidates(parsed_pdf):
    for picture in parsed_pdf.pictures:
        assert isinstance(picture.item, PictureItem)
        assert picture.page_number >= 1
        assert picture.element_index >= 0

    picture_indexes = [
        (picture.page_number, picture.element_index)
        for picture in parsed_pdf.pictures
    ]

    assert len(picture_indexes) == len(set(picture_indexes))


def test_records_convert_to_langchain_documents(parsed_pdf):
    for record in parsed_pdf.records:
        document = record.to_langchain_document()

        assert document.page_content == record.content
        assert document.metadata["record_id"] == record.record_id
        assert document.metadata["document_id"] == record.document_id


def test_missing_file_raises_error(parser):
    missing_file = PROJECT_ROOT / "missing_test_file.pdf"

    with pytest.raises(FileNotFoundError):
        parser.parse_file(
            str(missing_file),
            "missing_document",
        )


def test_non_pdf_file_raises_error(parser, tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        parser.parse_file(
            str(test_file),
            "invalid_document",
        )
