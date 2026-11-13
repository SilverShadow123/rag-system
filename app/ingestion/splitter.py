from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.ingestion.records import IngestionRecord

def recursive_splitter(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators =["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def split_records(
    records: list[IngestionRecord],
) -> list[IngestionRecord]: