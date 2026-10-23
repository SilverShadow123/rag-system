import re
from pathlib import Path

import pytest

from app.ingestion.loader import compute_document_id, load_documents, load_pdf
from app.ingestion.splitter import recursive_splitter

SAMPLE_PDF = Path("docs/thesis.pdf")


def test_load_and_split_documents():
    documents = load_documents("docs")
    chunks = recursive_splitter(documents)

    assert len(documents) > 0
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata


def test_load_pdf_returns_content_with_normalized_metadata():
    pages = load_pdf(SAMPLE_PDF)

    assert pages

    for page in pages:
        assert page.metadata["source"] == SAMPLE_PDF.name
        assert page.metadata["document_id"]
        assert page.page_content.strip()

    numbers = [page.metadata["page"] for page in pages]
    assert numbers[0] == 1
    assert numbers == sorted(numbers)


def test_load_pdf_document_id_matches_compute_document_id():
    pages = load_pdf(SAMPLE_PDF)

    assert pages[0].metadata["document_id"] == compute_document_id(SAMPLE_PDF)


def test_load_pdf_collapses_pdf_line_wraps():
    for page in load_pdf(SAMPLE_PDF):
        assert re.search(r"(?<!\n)\n(?!\n)", page.page_content) is None


def test_load_pdf_removes_standalone_page_numbers():
    for page in load_pdf(SAMPLE_PDF):
        lines = {line.strip() for line in page.page_content.splitlines()}
        assert str(page.metadata["page"]) not in lines


def test_load_pdf_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    with pytest.raises(ValueError):
        load_pdf(path)


def test_load_documents_ignores_non_pdf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "valid.pdf").write_bytes(SAMPLE_PDF.read_bytes())

    documents = load_documents(tmp_path)

    assert documents
    assert all(document.metadata["source"] == "valid.pdf" for document in documents)


def test_load_documents_isolates_broken_pdfs(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a real pdf")
    (tmp_path / "valid.pdf").write_bytes(SAMPLE_PDF.read_bytes())

    documents = load_documents(tmp_path)

    assert documents
    assert all(document.metadata["source"] == "valid.pdf" for document in documents)


def test_load_documents_returns_empty_for_folder_without_pdfs(tmp_path):
    assert load_documents(tmp_path) == []


def test_load_documents_raises_for_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_documents(tmp_path / "missing")


def test_load_documents_processes_files_in_sorted_order(tmp_path):
    (tmp_path / "b.pdf").write_bytes(SAMPLE_PDF.read_bytes())
    (tmp_path / "A.pdf").write_bytes(SAMPLE_PDF.read_bytes())

    sources = [document.metadata["source"] for document in load_documents(tmp_path)]

    assert sources[0] == "A.pdf"
    assert "b.pdf" in sources
