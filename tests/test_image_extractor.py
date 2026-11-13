
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from app.ingestion.docling_parser import DoclingParser
from app.ingestion.image_extractor import ImageExtractor


# Use the same PDF you used in test_docling_parser.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = PROJECT_ROOT / "docs" / "2401.01473v3.pdf"


@pytest.fixture(scope="module")
def parsed_pdf():
    """Convert the sample PDF and prepare picture candidates."""

    if not SAMPLE_PDF.is_file():
        pytest.fail(f"Test PDF not found: {SAMPLE_PDF}")

    parser = DoclingParser()

    return parser.parse_file(
        file_path=str(SAMPLE_PDF),
        document_id="test_document_001",
    )


@pytest.fixture
def extractor(tmp_path):
    """Use a temporary image directory for every test."""

    return ImageExtractor(
        image_root=tmp_path / "images"
    )


def test_pdf_contains_picture_candidates(parsed_pdf):
    """Ensure the sample PDF actually contains detectable pictures."""

    assert len(parsed_pdf.pictures) > 0, (
        "No picture candidates found. "
        "Use a PDF containing figures or diagrams."
    )


def test_extract_images_returns_assets(parsed_pdf, extractor):
    """Verify that the extractor returns ImageAsset objects."""

    assets = extractor.extract_images(parsed_pdf)

    assert len(assets) > 0

    for asset in assets:
        assert asset.document_id == parsed_pdf.document_id
        assert asset.source == parsed_pdf.source
        assert asset.page_number >= 1
        assert asset.element_index >= 0
        assert asset.extractor == "docling"


def test_images_are_saved(parsed_pdf, extractor):
    """Verify that extracted image files exist and are not empty."""

    assets = extractor.extract_images(parsed_pdf)

    assert len(assets) > 0

    for asset in assets:
        image_path = Path(asset.image_path)

        assert image_path.is_file()
        assert image_path.stat().st_size > 0


def test_saved_images_are_valid_png(parsed_pdf, extractor):
    """Verify that saved files can actually be opened as PNG images."""

    assets = extractor.extract_images(parsed_pdf)

    assert len(assets) > 0

    for asset in assets:
        with Image.open(asset.image_path) as image:
            assert image.format == "PNG"
            assert image.width > 0
            assert image.height > 0


def test_image_filenames_follow_convention(parsed_pdf, extractor):
    """Verify deterministic image filenames."""

    assets = extractor.extract_images(parsed_pdf)

    assert len(assets) > 0

    for asset in assets:
        expected_filename = (
            f"page_{asset.page_number:04d}"
            f"_picture_{asset.element_index:03d}.png"
        )

        assert Path(asset.image_path).name == expected_filename


def test_asset_ids_are_unique(parsed_pdf, extractor):
    """Ensure that every extracted image has a unique asset ID."""

    assets = extractor.extract_images(parsed_pdf)

    asset_ids = [
        asset.asset_id
        for asset in assets
    ]

    assert len(asset_ids) == len(set(asset_ids))


def test_image_indexes_match_picture_candidates(parsed_pdf, extractor):
    """Verify that saved images retain their original picture indexes."""

    assets = extractor.extract_images(parsed_pdf)

    candidate_indexes = {
        (picture.page_number, picture.element_index)
        for picture in parsed_pdf.pictures
    }

    for asset in assets:
        assert (
            asset.page_number,
            asset.element_index,
        ) in candidate_indexes


def test_empty_picture_list_returns_empty_assets(parsed_pdf, extractor):
    """Verify that a PDF without picture candidates returns an empty list."""

    parsed_without_pictures = replace(
        parsed_pdf,
        pictures=[],
    )

    assets = extractor.extract_images(parsed_without_pictures)

    assert assets == []
