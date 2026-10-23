from langchain_core.documents import Document

from app.ingestion.splitter import recursive_splitter


def _page(content: str, page: int, document_id: str = "doc-1", source: str = "doc.pdf") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "document_id": document_id, "page": page},
    )


def test_empty_input_returns_empty_list():
    assert recursive_splitter([]) == []


def test_short_document_returns_single_chunk():
    chunks = recursive_splitter([_page("Hello world. This is short.", 1)])

    assert len(chunks) == 1
    assert "Hello world" in chunks[0].page_content


def test_chunks_never_exceed_configured_chunk_size():
    text = " ".join(f"token{index:04d}" for index in range(2000))

    chunks = recursive_splitter([_page(text, 1)])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_consecutive_chunks_overlap_within_configured_window():
    text = " ".join(f"token{index:04d}" for index in range(1500))

    chunks = recursive_splitter([_page(text, 1)])

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        left = previous.page_content
        right = current.page_content
        overlap = 0
        for size in range(1, min(len(left), len(right)) + 1):
            if left[-size:] == right[:size]:
                overlap = size
        assert 100 <= overlap <= 200


def test_chunks_inherit_document_metadata():
    pages = [
        _page(
            "Sentence one. Sentence two. Sentence three.",
            1,
            document_id="abc",
            source="file.pdf",
        )
    ]

    chunks = recursive_splitter(pages)

    for chunk in chunks:
        assert chunk.metadata["document_id"] == "abc"
        assert chunk.metadata["source"] == "file.pdf"
        assert chunk.metadata["page"] == 1


def test_pages_of_same_document_are_joined_before_splitting():
    pages = [
        _page("The quick brown fox.", 1),
        _page("The lazy dog sleeps.", 2),
    ]

    chunks = recursive_splitter(pages)

    assert len(chunks) == 1
    assert "quick brown fox" in chunks[0].page_content
    assert "lazy dog" in chunks[0].page_content


def test_chunks_do_not_cross_document_boundaries():
    pages = [
        _page("Alpha content lives here.", 1, document_id="A", source="a.pdf"),
        _page("Beta content lives here.", 1, document_id="B", source="b.pdf"),
    ]

    chunks = recursive_splitter(pages)

    assert len(chunks) == 2

    by_document = {chunk.metadata["document_id"]: chunk.page_content for chunk in chunks}
    assert "Alpha" in by_document["A"]
    assert "Beta" not in by_document["A"]
    assert "Beta" in by_document["B"]
    assert "Alpha" not in by_document["B"]


def test_chunk_page_metadata_covers_all_source_pages():
    page_one = " ".join(f"alpha{index:04d}" for index in range(500))
    page_two = " ".join(f"beta{index:04d}" for index in range(500))

    chunks = recursive_splitter([_page(page_one, 1), _page(page_two, 2)])

    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}


def test_chunk_starting_on_second_page_reports_second_page():
    page_one = " ".join(f"alpha{index:04d}" for index in range(500))
    page_two = " ".join(f"beta{index:04d}" for index in range(500))

    chunks = recursive_splitter([_page(page_one, 1), _page(page_two, 2)])

    beta_chunks = [chunk for chunk in chunks if "beta" in chunk.page_content]
    assert beta_chunks
    assert all(chunk.metadata["page"] in {1, 2} for chunk in beta_chunks)
    assert any(chunk.metadata["page"] == 2 for chunk in beta_chunks)


def test_start_index_metadata_is_attached():
    text = " ".join(f"token{index:04d}" for index in range(1500))

    chunks = recursive_splitter([_page(text, 1)])

    assert all("start_index" in chunk.metadata for chunk in chunks)
    indices = [chunk.metadata["start_index"] for chunk in chunks]
    assert indices == sorted(indices)
