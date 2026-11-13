
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.ingestion.records import IngestionRecord


def recursive_splitter(
    documents: list[Document],
) -> list[Document]:

    if not documents:
        return []

    # Group pages belonging to the same PDF.
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
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    all_chunks = []

    for pages in grouped_documents.values():

        # Sort pages in their original PDF order.
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

        # Combine all pages belonging to this PDF.
        combined_content = page_separator.join(page_contents)

        combined_document = Document(
            page_content=combined_content,
            metadata=dict(pages[0].metadata),
        )

        # Split the combined document.
        chunks = splitter.split_documents(
            [combined_document]
        )

        # Restore the page number corresponding to each chunk's start.
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


def split_records(
    records: list[IngestionRecord],
) -> list[IngestionRecord]:
    raise NotImplementedError(
        "Record-aware splitting has not been implemented yet."
    )
