from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_documents(folder_path: str)-> list[Document]:
    documents: list[Document] = []

    for pdf_path in Path(folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path), extraction_mode='plain')
        documents.extend(loader.load())


    return documents