from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from collections import defaultdict
from dataclasses import dataclass
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)
from docling_core.types.doc import (
    DoclingDocument,
    TextItem,
    TableItem,
    PictureItem
)
import logging
logger = logging.getLogger(__name__)
from app.ingestion.records import (
    IngestionRecord,
    make_record_id
)

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_picture_images=True,
)


@dataclass
class PictureCandidate:
    item: PictureItem
    page_number: int
    element_index: int


@dataclass
class ParsedPDF:
    document_id: str
    source: str
    document: DoclingDocument
    records: list[IngestionRecord]
    pictures: list[PictureCandidate]
    markdown: str
    status: str


class DoclingParser:

    def __init__(self):
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse_file(self, file_path: str, document_id: str) -> ParsedPDF:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() != '.pdf':
            raise ValueError('Only PDF files are supported')

        result = self.converter.convert(path)

        document = result.document
        records: list[IngestionRecord] = []
        pictures: list[PictureCandidate] = []

        element_counters = defaultdict(int)

        for item, _level in document.iterate_items(
                traverse_pictures=True
        ):
            provenance = getattr(item, 'prov', None) or []

            if not provenance:
                logger.warning(
                    "Skipping %s in %s: no provenance",
                    type(item).__name__,
                    path.name,
                )
                continue

            page_number = provenance[0].page_no

            if isinstance(item, TextItem):
                element_type = 'text'
                content = item.text
            elif isinstance(item, TableItem):
                element_type = 'table'
                content = item.export_to_markdown(doc=document)
            elif isinstance(item, PictureItem):
                key = (page_number, 'picture')
                element_index = element_counters[key]
                element_counters[key] += 1

                candidate = PictureCandidate(
                    item=item,
                    page_number=page_number,
                    element_index=element_index,
                )
                pictures.append(candidate)
                continue
            else:
                continue

            content = content.strip()
            if not content:
                continue

            key = (page_number, element_type)
            element_index = element_counters[key]
            element_counters[key] += 1

            record = IngestionRecord(
                record_id=make_record_id(
                    document_id,
                    page_number,
                    element_type,
                    element_index
                ),
                document_id=document_id,
                source=path.name,
                page_number=page_number,
                element_type=element_type,
                element_index=element_index,
                content=content,
                extractor='docling'
            )
            records.append(record)

        return ParsedPDF(
            document_id=document_id,
            source=path.name,
            document=document,
            records=records,
            pictures=pictures,
            markdown=document.export_to_markdown(),
            status=str(result.status),
        )
