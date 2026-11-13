
import pytest

from app.ingestion.records import (
    IngestionRecord,
    make_record_id,
)
from app.ingestion.splitter import (
    CHUNK_SIZE,
    split_records,
)


def make_record(**overrides):

    data = {
        "record_id": "doc123-p1-text-e0-c0",
        "document_id": "doc123",
        "source": "sample.pdf",
        "page_number": 1,
        "element_type": "text",
        "element_index": 0,
        "chunk_index": 0,
        "content": "Sample content.",
        "extractor": "docling",
    }

    data.update(overrides)

    return IngestionRecord.model_validate(data)


def test_empty_records():

    assert split_records([]) == []


def test_short_text_record_is_preserved():

    record = make_record()

    chunks = split_records([record])

    assert len(chunks) == 1
    assert chunks[0] == record


def test_long_text_record_is_split():

    content = " ".join(
        f"word{index}"
        for index in range(1000)
    )

    record = make_record(content=content)

    chunks = split_records([record])

    assert len(chunks) > 1

    assert all(
        len(chunk.content) <= CHUNK_SIZE
        for chunk in chunks
    )


def test_text_chunk_indexes_are_sequential():

    content = " ".join(
        f"word{index}"
        for index in range(1000)
    )

    record = make_record(content=content)

    chunks = split_records([record])

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_text_chunk_ids_are_deterministic():

    content = " ".join(
        f"word{index}"
        for index in range(1000)
    )

    record = make_record(content=content)

    first = split_records([record])
    second = split_records([record])

    assert [
        chunk.record_id for chunk in first
    ] == [
        chunk.record_id for chunk in second
    ]


def test_text_chunks_preserve_metadata():

    content = " ".join(
        f"word{index}"
        for index in range(1000)
    )

    record = make_record(content=content)

    chunks = split_records([record])

    for chunk in chunks:
        assert chunk.document_id == record.document_id
        assert chunk.source == record.source
        assert chunk.page_number == record.page_number
        assert chunk.element_index == record.element_index
        assert chunk.extractor == record.extractor


def test_large_table_is_split_by_rows():

    rows = [
        f"| Model {index} | {index * 10}% | "
        f"{'experimental result ' * 3}|"
        for index in range(80)
    ]

    content = "\n".join([
        "| Model | Accuracy | Description |",
        "|---|---|---|",
        *rows,
    ])

    record = make_record(
        element_type="table",
        content=content,
    )

    chunks = split_records([record])

    assert len(chunks) > 1

    for chunk in chunks:
        lines = chunk.content.splitlines()

        assert lines[0] == "| Model | Accuracy | Description |"
        assert lines[1] == "|---|---|---|"
        assert len(lines) >= 3


def test_table_chunk_indexes_are_sequential():

    rows = [
        f"| Model {index} | {'result ' * 10}|"
        for index in range(80)
    ]

    content = "\n".join([
        "| Model | Description |",
        "|---|---|",
        *rows,
    ])

    record = make_record(
        element_type="table",
        content=content,
    )

    chunks = split_records([record])

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_small_table_is_preserved():

    content = (
        "| Model | Accuracy |\n"
        "|---|---|\n"
        "| A | 92% |"
    )

    record = make_record(
        element_type="table",
        content=content,
    )

    chunks = split_records([record])

    assert len(chunks) == 1
    assert chunks[0].content == content


def test_image_description_is_not_split():

    record = make_record(
        element_type="image_description",
        content="A neural network diagram. " * 100,
        image_path="data/images/doc123/page_0001_picture_000.png",
    )

    chunks = split_records([record])

    assert len(chunks) == 1

    assert chunks[0].content == record.content

    assert chunks[0].image_path == record.image_path


def test_image_description_keeps_original_id():

    record = make_record(
        element_type="image_description",
        content="A diagram showing a neural network.",
        image_path="data/images/doc123/page_0001_picture_000.png",
    )

    chunks = split_records([record])

    assert chunks[0].record_id == record.record_id


def test_multiple_record_types_are_processed():

    text_record = make_record(
        content="A short paragraph."
    )

    table_record = make_record(
        element_type="table",
        element_index=0,
        record_id="doc123-p1-table-e0-c0",
        content=(
            "| Model | Score |\n"
            "|---|---|\n"
            "| A | 90 |"
        ),
    )

    image_record = make_record(
        element_type="image_description",
        record_id="doc123-p1-image_description-e0-c0",
        content="A neural network diagram.",
        image_path="data/images/doc123/image.png",
    )

    chunks = split_records([
        text_record,
        table_record,
        image_record,
    ])

    assert len(chunks) == 3

    assert [
        chunk.element_type for chunk in chunks
    ] == [
        "text",
        "table",
        "image_description",
    ]
