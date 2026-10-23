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

## What each metric means

| Metric | Type | Meaning |
|---|---|---|
| Precision@k | Retrieval | Of the k chunks retrieved, what fraction are actually relevant. High means little noise. |
| Recall@k | Retrieval | Of all the relevant chunks that exist, what fraction showed up in the top k. High means little is missed. |
| Hit Rate@k | Retrieval | Fraction of questions where at least one relevant chunk appears in the top k. |
| MRR@k | Retrieval | Mean Reciprocal Rank. For each question, 1 divided by the rank of the first relevant chunk (1st = 1, 2nd = 0.5, 3rd = 0.33, none = 0), averaged over all questions. High means relevant chunks are ranked near the top. |
| Faithfulness | Generation (RAGAS) | Are the claims in the answer supported by the retrieved context. Low means the model is hallucinating. |
| Answer Relevancy | Generation (RAGAS) | How directly the answer addresses the question, with no off-topic or incomplete content. |
| Context Precision | Retrieval (RAGAS) | Whether the useful chunks are ranked above the useless ones in the retrieved context. |
| Context Recall | Retrieval (RAGAS) | How much of the information in the ground-truth answer is covered by the retrieved context. |

The @1, @3, @5 suffix means only the top 1, 3, or 5 retrieved chunks are considered.
