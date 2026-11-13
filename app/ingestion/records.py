from typing import Literal

from pydantic import (BaseModel, ConfigDict, Field, field_validator, model_validator)
from langchain_core.documents import Document

RecordType = Literal["text", "table", "image_description"]


class IngestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)

    page_number: int = Field(ge=1)
    element_type: RecordType
    element_index: int = Field(ge=0, description='Zero-based index of this element within its PDF page and element type')
    chunk_index: int = Field(default=0, ge=0)

    content: str = Field(min_length=1)
    extractor: str = Field(min_length=1)

    image_path: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Record content cannot be empty.")

        return value

    @model_validator(mode='after')
    def validate_image_path(self) -> "IngestionRecord":
        if self.element_type == 'image_description' and not self.image_path:
            raise ValueError(
                'Image descriptions must reference an original image'
            )
        if self.element_type in ('text', 'table') and self.image_path is not None:
            raise ValueError(
                'Text and table records should not have an image path'
            )
        return self

    def to_langchain_document(self) -> Document:
        metadata = {
            "record_id": self.record_id,
            "document_id": self.document_id,
            "source": self.source,
            "page_number": self.page_number,
            "element_type": self.element_type,
            "element_index": self.element_index,
            "chunk_index": self.chunk_index,
            "extractor": self.extractor,
        }

        if self.image_path is not None:
            metadata["image_path"] = self.image_path

        return Document(
            page_content=self.content,
            metadata=metadata,
        )

class ImageAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)

    page_number: int = Field(ge=1)
    element_index: int = Field(ge=0)

    image_path: str = Field(min_length=1)
    extractor: str = 'docling'


def make_record_id(
        document_id: str,
        page_number: int,
        element_type: RecordType,
        element_index: int,
        chunk_index: int = 0,
) -> str:
    return (
        f"{document_id}-p{page_number}"
        f"-{element_type}-e{element_index}-c{chunk_index}"
    )
