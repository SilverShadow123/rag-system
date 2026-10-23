
"""LLM-based RAG evaluation using Ragas."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate generated responses and retrieved contexts with Ragas."""

    usable = [
        case
        for case in cases
        if case.get("response") is not None
        and isinstance(case.get("retrieved_contexts"), list)
        and case.get("retrieved_contexts")
    ]

    if not usable:
        raise ValueError(
            "No successful cases with retrieved contexts "
            "are available for Ragas evaluation."
        )

    try:
        from ragas import evaluate
        from ragas.dataset_schema import (
            EvaluationDataset,
            SingleTurnSample,
        )
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the expected Ragas API. "
            "Check your installed Ragas version and metric API. "
            f"Original error: {exc}"
        ) from exc

    try:
        from app.generation.generator import llm
        from app.retrieval.vector_store import create_embeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(
            create_embeddings()
        )

        samples = []

        for case in usable:
            samples.append(
                SingleTurnSample(
                    user_input=case["user_input"],
                    retrieved_contexts=case["retrieved_contexts"],
                    response=case["response"],
                    reference=case.get("reference"),
                    reference_contexts=case.get(
                        "reference_contexts"
                    ),
                )
            )

        dataset = EvaluationDataset(samples=samples)

        metrics = [
            Faithfulness(llm=wrapped_llm),
            AnswerRelevancy(
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
            ),
            ContextPrecision(llm=wrapped_llm),
            ContextRecall(llm=wrapped_llm),
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        frame = result.to_pandas()

        excluded_columns = {
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
            "reference_contexts",
        }

        aggregate = {}

        for column in frame.columns:
            if column in excluded_columns:
                continue

            values = frame[column].dropna()

            aggregate[column] = (
                float(values.mean())
                if len(values)
                else None
            )

        return {
            "status": "success",
            "case_count": len(usable),
            "aggregate": aggregate,
            "scores": frame.to_dict(orient="records"),
        }

    except Exception as exc:
        logger.exception("Ragas evaluation failed")

        raise RuntimeError(
            "Ragas scoring failed. Check the installed Ragas API, "
            "metric compatibility, model access, and embeddings. "
            f"Original error: {exc}"
        ) from exc
