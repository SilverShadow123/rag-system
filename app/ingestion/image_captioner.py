
import base64
import logging
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.ingestion.records import (
    ImageAsset,
    IngestionRecord,
    make_record_id,
)


logger = logging.getLogger(__name__)


CAPTION_PROMPT = """
You are analyzing an image extracted from an academic PDF.

Describe the image accurately and objectively.

Identify:

1. The type of visual: chart, diagram, graph, photograph, etc.
2. Its main subject and purpose.
3. Important labels, components and visible text.
4. Relationships and connections between components.
5. Important numerical values when they are legible.

For charts, describe axes, trends and comparisons.

For architecture diagrams, explain the components and
the direction of information flow.

For tables embedded in images, preserve visible headers
and values.

Do not invent missing information.
Do not guess illegible labels or numbers.
If something cannot be determined, say so.

Return a concise but informative description.
"""


SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


class ImageCaptioner:

    def __init__(self, model: BaseChatModel):
        self.model = model

    def _encode_image(self, image_path: Path) -> tuple[str, str]:

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(image_path.name)

        if mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image format: {mime_type}"
            )

        image_bytes = image_path.read_bytes()

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        return encoded_image, mime_type

    @staticmethod
    def _extract_response_text(response) -> str:

        content = response.content

        if isinstance(content, str):
            description = content

        elif isinstance(content, list):
            text_parts = []

            for block in content:

                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    if block.get("type") in ("text", "output_text"):
                        text = block.get("text")

                        if isinstance(text, str):
                            text_parts.append(text)

            description = "\n".join(text_parts)

        else:
            raise ValueError(
                "Vision model returned an unsupported response format."
            )

        description = description.strip()

        if not description:
            raise ValueError(
                "Vision model returned an empty description."
            )

        return description

    def describe_asset(
        self,
        asset: ImageAsset,
    ) -> IngestionRecord:

        image_path = Path(asset.image_path)

        encoded_image, mime_type = self._encode_image(
            image_path
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": CAPTION_PROMPT,
                },
                {
                    "type": "image",
                    "base64": encoded_image,
                    "mime_type": mime_type,
                },
            ]
        )

        try:
            response = self.model.invoke([message])

        except Exception:
            logger.exception(
                "Caption generation failed: document=%s, page=%s, image=%s",
                asset.document_id,
                asset.page_number,
                asset.element_index,
            )
            raise

        description = self._extract_response_text(response)

        record = IngestionRecord(
            record_id=make_record_id(
                asset.document_id,
                asset.page_number,
                "image_description",
                asset.element_index,
            ),
            document_id=asset.document_id,
            source=asset.source,
            page_number=asset.page_number,
            element_type="image_description",
            element_index=asset.element_index,
            content=description,
            extractor="vision_model",
            image_path=asset.image_path,
        )

        return record

    def describe_assets(
        self,
        assets: list[ImageAsset],
    ) -> list[IngestionRecord]:

        records = []

        for asset in assets:
            record = self.describe_asset(asset)
            records.append(record)

        return records