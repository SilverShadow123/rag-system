from app.ingestion.loader import load_documents
from app.ingestion.splitter import recursive_splitter
from langchain_core.documents import Document

def ingest_documents(folder_path: str) -> list[Document]:
    documents = load_documents(folder_path)
    chunks = recursive_splitter(documents)
    return chunks