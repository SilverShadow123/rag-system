
import logging
import re
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.ingestion.records import (
    IngestionRecord,
    make_record_id,
)


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def recursive_splitter(
    documents: list[Document],
) -> list[Document]:

    if not documents:
        return []

    grouped_documents = defaultdict(list)

    for index, document in enumerate(documents):

        document_id = document.metadata.get("document_id")
        source = document.metadata.get("source")

        if document_id is None and source is None:
            key = ("unknown", index)
        else:
            key = (document_id, source)

        grouped_documents[key].append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    all_chunks = []

    for pages in grouped_documents.values():

        pages.sort(
            key=lambda page: page.metadata.get("page", 0)
        )

        page_offsets = []
        page_contents = []

        current_offset = 0
        page_separator = "\n\n"

        for page in pages:

            content = page.page_content.strip()

            if not content:
                continue

            if page_contents:
                current_offset += len(page_separator)

            page_number = page.metadata.get("page")

            page_offsets.append(
                (current_offset, page_number)
            )

            page_contents.append(content)

            current_offset += len(content)

        if not page_contents:
            continue

        combined_content = page_separator.join(page_contents)

        combined_document = Document(
            page_content=combined_content,
            metadata=dict(pages[0].metadata),
        )

        chunks = splitter.split_documents(
            [combined_document]
        )

        for chunk in chunks:

            start_index = chunk.metadata["start_index"]

            starting_page = page_offsets[0][1]

            for page_start, page_number in page_offsets:

                if page_start <= start_index:
                    starting_page = page_number
                else:
                    break

            chunk.metadata["page"] = starting_page

        all_chunks.extend(chunks)

    return all_chunks


def _create_chunk_record(
    record: IngestionRecord,
    content: str,
    chunk_index: int,
) -> IngestionRecord:

    data = record.model_dump()

    data.update({
        "record_id": make_record_id(
            record.document_id,
            record.page_number,
            record.element_type,
            record.element_index,
            chunk_index,
        ),
        "chunk_index": chunk_index,
        "content": content,
    })

    return IngestionRecord.model_validate(data)

def _merge_adjacent_text_records(
    records: list[IngestionRecord],
) -> list[IngestionRecord]:

    merged_records = []
    pending_text_records = []

    def flush_text_group():
        nonlocal pending_text_records

        if not pending_text_records:
            return

        first_record = pending_text_records[0]

        if len(pending_text_records) == 1:
            merged_records.append(first_record)
        else:
            combined_content = "\n\n".join(
                record.content
                for record in pending_text_records
            )

            merged_record = first_record.model_copy(
                update={
                    "record_id": make_record_id(
                        first_record.document_id,
                        first_record.page_number,
                        "text",
                        first_record.element_index,
                        0,
                    ),
                    "chunk_index": 0,
                    "content": combined_content,
                }
            )

            merged_records.append(merged_record)

        pending_text_records = []

    for record in records:

        # Tables and image descriptions break text groups.
        if record.element_type != "text":
            flush_text_group()
            merged_records.append(record)
            continue

        # Only merge records from the same page and document.
        if pending_text_records:

            previous = pending_text_records[-1]

            same_page = (
                previous.document_id == record.document_id
                and previous.source == record.source
                and previous.page_number == record.page_number
                and previous.extractor == record.extractor
            )

            if not same_page:
                flush_text_group()

        pending_text_records.append(record)

    flush_text_group()

    return merged_records


def _split_text_record(
    record: IngestionRecord,
    splitter: RecursiveCharacterTextSplitter,
) -> list[IngestionRecord]:

    if len(record.content) <= CHUNK_SIZE:
        return [record]

    chunks = splitter.split_text(record.content)

    return [
        _create_chunk_record(
            record=record,
            content=content,
            chunk_index=index,
        )
        for index, content in enumerate(chunks)
        if content.strip()
    ]


def _is_markdown_separator(line: str) -> bool:

    cells = line.strip().strip("|").split("|")

    return bool(cells) and all(
        re.fullmatch(r"\s*:?-{3,}:?\s*", cell)
        for cell in cells
    )


def _split_table_record(
    record: IngestionRecord,
) -> list[IngestionRecord]:

    if len(record.content) <= CHUNK_SIZE:
        return [record]

    lines = [
        line.strip()
        for line in record.content.splitlines()
        if line.strip()
    ]

    # A Markdown table needs a header and separator row.
    if len(lines) < 2 or not _is_markdown_separator(lines[1]):

        logger.warning(
            "Cannot safely split non-standard Markdown table: %s",
            record.record_id,
        )

        return [record]

    header = lines[:2]
    data_rows = lines[2:]

    if not data_rows:
        return [record]

    table_chunks = []
    current_rows = []

    for row in data_rows:

        candidate_content = "\n".join(
            header + current_rows + [row]
        )

        if len(candidate_content) > CHUNK_SIZE and current_rows:

            table_chunks.append(
                "\n".join(header + current_rows)
            )

            current_rows = [row]

        else:
            current_rows.append(row)

        # A single row may exceed CHUNK_SIZE.
        # Keep it intact rather than destroying its structure.
        if len("\n".join(header + current_rows)) > CHUNK_SIZE:
            logger.warning(
                "A table row exceeds chunk size: %s",
                record.record_id,
            )

    if current_rows:
        table_chunks.append(
            "\n".join(header + current_rows)
        )

    return [
        _create_chunk_record(
            record=record,
            content=content,
            chunk_index=index,
        )
        for index, content in enumerate(table_chunks)
    ]


def split_records(
    records: list[IngestionRecord],
) -> list[IngestionRecord]:

    if not records:
        return []


    records = _merge_adjacent_text_records(records)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_records = []

    for record in records:

        if record.element_type == "text":

            chunks = _split_text_record(
                record,
                text_splitter,
            )

        elif record.element_type == "table":

            chunks = _split_table_record(record)

        elif record.element_type == "image_description":

            chunks = [record]

        else:
            raise ValueError(
                f"Unsupported record type: {record.element_type}"
            )

        chunked_records.extend(chunks)

    return chunked_records
