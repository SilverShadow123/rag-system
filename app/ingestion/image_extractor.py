import logging
from pathlib import Path
from app.ingestion.docling_parser import ParsedPDF
from app.ingestion.records import ImageAsset

logger = logging.getLogger(__name__)


class ImageExtractor:

    def __init__(
            self,
            image_root: str | Path = 'data/images',
    ):
        self.image_root = Path(image_root)

    def extract_images(self, parsed: ParsedPDF) -> list[ImageAsset]:
        assets: list[ImageAsset] = []

        document_dir = self.image_root / parsed.document_id

        for candidate in parsed.pictures:
            image = candidate.item.get_image(parsed.document)

            if image is None:
                logger.warning(
                    "Image missing: document=%s, page=%s, element=%s",
                    parsed.document_id,
                    candidate.page_number,
                    candidate.element_index,
                )
                continue

            filename = (
                f"page_{candidate.page_number:04d}"
                f"_picture_{candidate.element_index:03d}.png"
            )

            image_path = document_dir / filename

            image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image.save(image_path, format="PNG")

            asset = ImageAsset(
                asset_id=(
                    f"{parsed.document_id}"
                    f"-p{candidate.page_number}"
                    f"-picture-e{candidate.element_index}"
                ),
                document_id=parsed.document_id,
                source=parsed.source,
                page_number=candidate.page_number,
                element_index=candidate.element_index,
                image_path=image_path.as_posix(),
                extractor="docling",
            )

            assets.append(asset)

        return assets


