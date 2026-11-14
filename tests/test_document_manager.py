
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.ingestion import document_manager


@pytest.fixture
def sample_pdf(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"test pdf content")
    return pdf


def test_add_document_uses_multimodal_pipeline(
    sample_pdf,
    monkeypatch,
):

    pipeline_mock = MagicMock(return_value=42)

    monkeypatch.setattr(
        document_manager,
        "ingest_and_index_multimodal_pdf",
        pipeline_mock,
    )

    count = document_manager.add_document(
        str(sample_pdf)
    )

    assert count == 42

    pipeline_mock.assert_called_once_with(
        str(sample_pdf)
    )


def test_add_document_rejects_missing_file(
    tmp_path,
):

    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        document_manager.add_document(
            str(missing_pdf)
        )


def test_add_document_rejects_unsupported_file(
    tmp_path,
):

    text_file = tmp_path / "sample.txt"
    text_file.write_text("sample")

    with pytest.raises(ValueError):
        document_manager.add_document(
            str(text_file)
        )


def test_add_document_propagates_pipeline_failure(
    sample_pdf,
    monkeypatch,
):

    pipeline_mock = MagicMock(
        side_effect=RuntimeError("Captioning failed")
    )

    monkeypatch.setattr(
        document_manager,
        "ingest_and_index_multimodal_pdf",
        pipeline_mock,
    )

    with pytest.raises(RuntimeError, match="Captioning failed"):
        document_manager.add_document(
            str(sample_pdf)
        )


def test_delete_document_removes_all_assets(
    sample_pdf,
    tmp_path,
    monkeypatch,
):

    document_id = "test-document-123"

    store = MagicMock()

    monkeypatch.setattr(
        document_manager,
        "compute_document_id",
        lambda path: document_id,
    )

    monkeypatch.setattr(
        document_manager,
        "load_vector_store",
        lambda: store,
    )

    image_root = tmp_path / "images"

    image_dir = image_root / document_id
    image_dir.mkdir(parents=True)

    image_file = image_dir / "figure.png"
    image_file.write_bytes(b"test image")

    monkeypatch.setattr(
        document_manager,
        "IMAGE_ROOT",
        image_root,
    )

    document_manager.delete_document(
        str(sample_pdf)
    )

    store.delete.assert_called_once_with(
        where={
            "document_id": document_id,
        }
    )

    assert not image_dir.exists()
    assert not sample_pdf.exists()


def test_delete_document_rejects_missing_file(
    tmp_path,
):

    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        document_manager.delete_document(
            str(missing_pdf)
        )
