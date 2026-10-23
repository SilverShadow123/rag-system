"""
Generate a synthetic RAG evaluation test set with Ragas.

No langchain_community needed. Ragas 0.4.3 imports it for two unused VertexAI
names, so run patch_ragas.py once after installing (see that file).

Install:
    pip install ragas pypdf python-dotenv langchain-huggingface langchain-openrouter
    pip uninstall -y langchain-community
    python patch_ragas.py

Run from the project root:
    python -m evals.generate_testset
"""
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from pypdf import PdfReader
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

load_dotenv()

DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("evals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TESTSET_SIZE = int(os.getenv("TESTSET_SIZE", "10"))
MIN_CHARS = 3000  # ~700 tokens; Ragas needs > 500 tokens per document
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(docs_dir: Path) -> list[Document]:
    """One Document per PDF (all pages merged). Avoids Ragas' 'headlines' error."""
    docs, short = [], []
    for pdf_path in sorted(docs_dir.rglob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        texts = [(p.extract_text() or "").strip() for p in reader.pages]
        text = "\n\n".join(t for t in texts if t)
        if not text:
            print(f"  Skipping {pdf_path.name}: no extractable text (scanned?).")
            continue
        doc = Document(page_content=text, metadata={"source": str(pdf_path)})
        if len(text) >= MIN_CHARS:
            docs.append(doc)
        else:
            print(f"  {pdf_path.name} is short ({len(text)} chars).")
            short.append(doc)

    # Fallback: combine short PDFs into a single document instead of crashing.
    if not docs and short:
        print("  Combining all short PDFs into one document.")
        docs.append(
            Document(
                page_content="\n\n".join(d.page_content for d in short),
                metadata={"source": "combined"},
            )
        )
    return docs


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is missing from .env")
    model_name = os.getenv("OPENROUTER_MODEL")
    if not model_name:
        raise ValueError("OPENROUTER_MODEL is missing from .env")
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {DOCS_DIR.resolve()}")

    docs = load_pdfs(DOCS_DIR)
    if not docs:
        raise ValueError("No readable PDF text found. Scanned PDFs need OCR first.")
    print(f"Using {len(docs)} document(s), "
          f"{sum(len(d.page_content) for d in docs)} characters.")

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(
            ChatOpenRouter(model=model_name, temperature=0, max_retries=2)
        ),
        embedding_model=LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        ),
    )

    print(f"Generating {TESTSET_SIZE} test cases. This may take a few minutes...")
    try:
        testset = generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=TESTSET_SIZE,
            run_config=RunConfig(max_workers=4, timeout=180, max_retries=5),
        )
    except ValueError as e:
        if "headlines" in str(e):
            raise SystemExit(
                "\nRagas could not split the documents (too little text). "
                "Add longer or more PDFs to the docs folder."
            ) from e
        raise

    df = testset.to_pandas()
    raw_path = OUTPUT_DIR / "synthetic_testset_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    clean = df.rename(
        columns={
            "user_input": "question",
            "reference": "ground_truth",
            "reference_contexts": "contexts",
        }
    )
    clean_path = OUTPUT_DIR / "synthetic_testset.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    print(f"\nGenerated {len(df)} test cases.")
    print(f"Raw   : {raw_path.resolve()}")
    print(f"Clean : {clean_path.resolve()}")
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()
