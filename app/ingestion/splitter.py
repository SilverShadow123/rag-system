from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def recursive_splitter(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators =["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)