# Intelligent RAG System

An evolving RAG engineering project. It starts with a naive RAG baseline for question answering over PDF documents and grows toward a reliable, evaluated, production-ready system. Every upgrade (hybrid retrieval, reranking, query rewriting, and so on) is measured against the same fixed test set before it is accepted.

Design principle: start with the smallest reliable baseline, keep components replaceable, and add complexity only when evaluation shows a clear benefit.

## Current status

| Component | Status |
|---|---|
| PDF ingestion, chunking, embeddings, ChromaDB | Done |
| Answer generation with source and page references (OpenRouter) | Done |
| Streamlit chat UI with PDF upload and delete | Done |
| Evaluation pipeline (classical metrics and RAGAS) | Done, baseline recorded |
| FastAPI backend | Only the /health endpoint so far |
| Hybrid retrieval (BM25 plus dense) | Planned |
| Reranking | Planned |
| Query rewriting | Planned |
| LangGraph workflows, Jev decision adapter | Planned |
| Flutter client, cloud vector store | Planned |

## How it works

Ingestion
1. A PDF is loaded page by page with PyPDFLoader. Empty pages are skipped.
2. Each document gets a stable SHA-256 document ID based on the file contents.
3. Pages are split with RecursiveCharacterTextSplitter (chunk size 1000, overlap 150).
4. Chunks are embedded with sentence-transformers/all-MiniLM-L6-v2 and stored in a local ChromaDB collection (data/chroma).
5. Chunk IDs are deterministic (document ID plus index), and previous chunks for the same document are deleted before re-adding, so re-ingestion does not create duplicates.

Query
1. The question is embedded and the top-k most similar chunks are retrieved from ChromaDB (k = 4 in the Streamlit app, k = 5 in the evals).
2. The chunks are passed as context, with source file and page, to an LLM through OpenRouter (temperature 0).
3. The prompt tells the model to answer only from the context, say so when the documents do not contain the answer, treat the context as reference material and not instructions, and cite the source document and page.
4. The Streamlit app shows the answer along with the retrieved source chunks.

## Tech stack

Python, FastAPI, LangChain, ChromaDB, Sentence Transformers (all-MiniLM-L6-v2), OpenRouter for the LLM, Streamlit for the temporary UI, RAGAS for LLM-based evaluation, pytest for tests.

## Project structure

```
rag-system/
├── app/
│   ├── api/            FastAPI app (health check)
│   ├── ingestion/      PDF loader, splitter, ingestion pipeline, document manager
│   ├── retrieval/      Embeddings, ChromaDB vector store, retriever
│   └── generation/     Prompt and OpenRouter answer generation
├── frontend/           Streamlit chat app
├── evals/              Test set generation, eval runner, metrics, saved results
├── docs/               Sample PDFs, evaluation plan, results table
├── tests/              pytest tests for loader, splitter, pipeline, retrieval, RAG flow
├── ARCHITECTURE.md     Target architecture and roadmap
└── requirements.txt
```

## Getting started

1. Create and activate a virtual environment, then install dependencies:

```
pip install -r requirements.txt
```

2. Create a .env file in the project root (it is excluded from Git):

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=your_chosen_model
```

3. Run the Streamlit app from the project root:

```
python -m streamlit run frontend/streamlit_app.py
```

Upload a PDF in the sidebar to index it, then ask questions in the chat.

4. Run the API (currently only the health check):

```
uvicorn app.api.main:app --reload
```

5. Run the tests:

```
pytest
```

## Running the evaluation

The eval uses a synthetic test set generated from the PDFs in docs/ with RAGAS.

```
python -m evals.generate_testset
python -m evals.run_eval
```

Notes:
- generate_testset needs OPENROUTER_API_KEY and OPENROUTER_MODEL, and the documents must contain enough text (RAGAS needs longer documents). The size defaults to 10 cases and can be changed with the TESTSET_SIZE environment variable.
- The generated CSV is converted to evals/synthetic_testset.json (see evals/convert_testset.py), which run_eval reads.
- If RAGAS fails to import because of langchain_community, evals/patch_ragas.py documents a one-time patch.
- run_eval retrieves the top 5 chunks per question, generates answers, then writes per-case outputs, classical_metrics.json, ragas_metrics.json, and summary.json to evals/results/.

## Evaluation results

Setup: 10 synthetic test cases, top_k = 5. Add a new column each time a new component is added.

| Metric | Naive RAG | Hybrid RAG | Hybrid + Reranking |
|---|---|---|---|
| Precision@1 | 0.100 | Not measured yet | Not measured yet |
| Recall@1 | 0.100 | Not measured yet | Not measured yet |
| Hit Rate@1 | 0.100 | Not measured yet | Not measured yet |
| MRR@1 | 0.100 | Not measured yet | Not measured yet |
| Precision@3 | 0.033 | Not measured yet | Not measured yet |
| Recall@3 | 0.100 | Not measured yet | Not measured yet |
| Hit Rate@3 | 0.100 | Not measured yet | Not measured yet |
| MRR@3 | 0.100 | Not measured yet | Not measured yet |
| Precision@5 | 0.060 | Not measured yet | Not measured yet |
| Recall@5 | 0.200 | Not measured yet | Not measured yet |
| Hit Rate@5 | 0.200 | Not measured yet | Not measured yet |
| MRR@5 | 0.120 | Not measured yet | Not measured yet |
| Faithfulness | 0.918 | Not measured yet | Not measured yet |
| Answer Relevancy | 0.890 | Not measured yet | Not measured yet |
| Context Precision | 0.849 | Not measured yet | Not measured yet |
| Context Recall | 0.875 | Not measured yet | Not measured yet |

Full metric definitions and the results table live in docs/rag_eval_results.md. The evaluation methodology and fair-comparison rules are in docs/EVALUATION_PLAN.md.

## What the baseline shows

Generation looks solid: faithfulness is 0.918 and answer relevancy is 0.890, so answers stay grounded in the retrieved context.

The classical retrieval scores are low (Recall@5 0.200, MRR@5 0.120), but they disagree with RAGAS context recall (0.875). The classical metrics use a heuristic normalized text containment check between retrieved chunks and the reference contexts, which is not the same as manually labeled relevance, and it can miss chunks that RAGAS judges relevant. Treat the classical numbers as a rough comparison signal for now, and consider labeling relevant chunks by hand or by chunk ID before drawing strong conclusions.

## Known limitations

- Only PDF files are supported.
- The classical retrieval metrics use a text containment heuristic, not hand-labeled relevance.
- The test set is small (10 synthetic cases), so scores are preliminary.
- The API does not yet expose document or query endpoints, and there is no authentication.
- The evaluation plan also calls for answer correctness, citation accuracy, abstention handling, latency, and error rate; these are not measured yet.

## Roadmap

1. Baseline RAG (done)
2. Reliability: deterministic IDs, validation, tests (in progress)
3. Evaluation baseline (done for the first run)
4. Retrieval improvements: tune chunking and top-k, add BM25 and hybrid fusion
5. Reranking and query rewriting
6. Decision workflows and LangGraph, only where they fix demonstrated failures
7. Production preparation: auth, deployment, observability, security
8. Flutter client on the same versioned API
9. Cloud vector store (Qdrant or Supabase with pgvector)

See ARCHITECTURE.md for the full design.