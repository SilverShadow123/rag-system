
import streamlit as st
from pathlib import Path

from app.retrieval.vector_store import load_vector_store, create_retriever
from app.generation.generator import generate_answer
from app.ingestion.document_manager import add_document, delete_document


st.set_page_config(
    page_title="Naive RAG Chatbot",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Naive RAG Chatbot")
st.caption("Ask questions about your indexed PDF documents.")


@st.cache_resource
def load_retriever():
    vector_store = load_vector_store()
    return create_retriever(vector_store, k=4)


retriever = load_retriever()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Settings")
    st.write("**Model:** OpenRouter")
    st.write("**Retrieval:** ChromaDB")
    st.write("**Embedding:** all-MiniLM-L6-v2")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Manage Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_upload",
    )

    if st.button("Add PDF", disabled=uploaded_file is None):
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)

        file_path = docs_dir / Path(uploaded_file.name).name

        if file_path.exists():
            st.warning(
                "A PDF with this filename already exists. "
                "Choose another filename or delete the existing document first."
            )
        else:
            try:
                file_path.write_bytes(uploaded_file.getvalue())

                chunk_count = add_document(str(file_path))

                load_retriever.clear()

                st.success(
                    f"Added {file_path.name}: {chunk_count} chunks indexed."
                )
                st.rerun()

            except Exception as error:
                file_path.unlink(missing_ok=True)
                st.error(f"Failed to add PDF: {error}")

    st.divider()
    st.subheader("Indexed Documents")

    docs_dir = Path("docs")
    pdf_files = sorted(docs_dir.glob("*.pdf")) if docs_dir.exists() else []

    if pdf_files:
        selected_file = st.selectbox(
            "Choose a PDF",
            options=pdf_files,
            format_func=lambda path: path.name,
            key="selected_pdf",
        )

        if st.button("Delete Selected PDF", type="secondary"):
            try:
                delete_document(str(selected_file))
                load_retriever.clear()

                st.success(f"Deleted {selected_file.name}.")
                st.rerun()

            except Exception as error:
                st.error(f"Failed to delete PDF: {error}")
    else:
        st.caption("No PDF files found in docs/.")


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"Retrieved source chunks ({len(message['sources'])})"
            ):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(
                        f"Source: {source['source']} | "
                        f"Page: {source['page']}"
                    )
                    st.write(source["content"])
                    st.divider()


# Chat input
if question := st.chat_input("Ask something about your documents..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get the latest retriever after document changes
                retriever = load_retriever()

                documents = retriever.invoke(question)
                answer = generate_answer(question, documents)

                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "content": doc.page_content,
                    }
                    for doc in documents
                ]

            except Exception:
                st.error(
                    "Something went wrong while processing your question. "
                    "Check the terminal logs and API configuration."
                )
                st.stop()

        st.markdown(answer)

        if sources:
            with st.expander(
                f"Retrieved source chunks ({len(sources)})"
            ):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(
                        f"Source: {source['source']} | "
                        f"Page: {source['page']}"
                    )
                    st.write(source["content"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
