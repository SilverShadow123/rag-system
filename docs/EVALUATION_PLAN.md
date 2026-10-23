# RAG System — Evaluation Plan

## 1. Purpose

This document defines how the initial naive RAG baseline will be evaluated and how later improvements will be compared. The first goal is a reliable, measurable baseline—not a claim of production-level performance.

## 2. Initial benchmark dataset

Create a small, fixed evaluation dataset using documents that we are permitted to use.

- Initial target: 30 questions.
- Each question should have an expected answer and, where possible, the relevant source document or chunk.
- Keep evaluation questions separate from any examples used to tune prompts.
- Store the dataset in `evals/benchmark.jsonl` (one JSON object per line).

Suggested question distribution:

| Question type | Target count | What it tests |
|---|---:|---|
| Direct factual | 10 | Answers explicitly stated in one passage |
| Multi-passage | 5 | Answers requiring information from multiple passages |
| Unsupported / unanswerable | 5 | Whether the system avoids inventing answers |
| Ambiguous | 5 | Whether the system asks for clarification or responds cautiously |
| Distracting / irrelevant context | 5 | Whether the system uses relevant evidence rather than similar-looking text |
| **Total** | **30** | |

These counts are targets, not results. If suitable documents are not yet available, start with 10–15 questions and expand to 30.

## 3. Metrics

| Metric | Meaning | How to evaluate initially |
|---|---|---|
| Retrieval Recall@5 | Fraction of questions for which a relevant source appears in the top five retrieved chunks | Mark relevant source IDs for each question; calculate the fraction retrieved |
| Answer correctness | Whether the response answers the question accurately | Manually score each answer as correct, partially correct, or incorrect |
| Faithfulness / grounding | Whether factual claims are supported by retrieved context | Check answer claims against the retrieved chunks |
| Citation accuracy | Whether cited sources actually support the claims | Check each citation against the source text |
| Abstention behavior | Whether the system avoids making up answers when evidence is insufficient | Review the unanswerable questions and count unsupported answers |
| Latency | Time taken to return an answer | Record elapsed time for each query; report median and p95 when the sample is large enough |
| Errors | Failed requests or unhandled exceptions | Record failures during benchmark runs |

For a small initial dataset, manual review is acceptable. Do not report an unmeasured metric as a result.

## 4. Baseline procedure

1. Use the same document collection and benchmark questions for each system version.
2. Save each question, retrieved chunk IDs, generated answer, citations, latency, and any error.
3. Score the answers using the definitions above.
4. Record the results before adding hybrid retrieval, reranking, query rewriting, or agentic workflows.
5. After each major change, rerun the same benchmark and compare results.
6. Investigate regressions instead of assuming a more complex system is better.

## 5. Initial result table

Complete this table after the first end-to-end RAG baseline works.

| Metric | Baseline result |
|---|---|
| Retrieval Recall@5 | Not measured yet |
| Answer correctness | Not measured yet |
| Faithfulness / grounding | Not measured yet |
| Citation accuracy | Not measured yet |
| Unsupported-question handling | Not measured yet |
| Median latency | Not measured yet |
| p95 latency | Not measured yet |
| Request error rate | Not measured yet |

## 6. Fair-comparison rules

- Keep the benchmark questions and source documents fixed when comparing versions.
- Record the model name, embedding model, chunking settings, retrieval `k`, prompt version, and run date.
- Do not tune the system against the benchmark and then treat the same benchmark as fully independent evidence.
- Treat small-sample scores as preliminary; manually inspect failures.
- Report limitations and actual measurements honestly.

