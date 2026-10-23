
"""Run retrieval, generation, and evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.generation.generator import generate_answer
from app.retrieval.vector_store import load_vector_store
from evals.classical_metrics import evaluate_retrieval_cases

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "synthetic_testset.json"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

TOP_K = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def load_testset(
    path: Path,
) -> list[dict[str, Any]]:
    """Load and validate the synthetic test dataset."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "synthetic_testset.json must contain a JSON list."
        )

    required_fields = {
        "user_input",
        "reference",
        "reference_contexts",
    }

    for index, case in enumerate(data, start=1):
        missing = required_fields - case.keys()

        if missing:
            raise ValueError(
                f"Case {index} is missing fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(case["reference_contexts"], list):
            raise ValueError(
                f"Case {index}: reference_contexts must be a list."
            )

        if (
            not isinstance(case["user_input"], str)
            or not case["user_input"].strip()
        ):
            raise ValueError(
                f"Case {index}: user_input must be non-empty."
            )

    return data


def run_pipeline(
    testset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retrieve evidence and generate answers for every test case."""

    vector_store = load_vector_store()
    results = []

    for index, case in enumerate(testset, start=1):
        question = case["user_input"]

        logger.info(
            "Evaluating question %d/%d",
            index,
            len(testset),
        )

        base = {
            "user_input": question,
            "reference": case.get("reference"),
            "reference_contexts": case.get(
                "reference_contexts", []
            ),
            "persona_name": case.get("persona_name"),
            "query_style": case.get("query_style"),
            "query_length": case.get("query_length"),
        }

        try:
            documents = vector_store.similarity_search(
                question,
                k=TOP_K,
            )

            retrieved_contexts = [
                document.page_content
                for document in documents
            ]

            retrieved_sources = [
                {
                    "source": document.metadata.get("source"),
                    "page": document.metadata.get("page"),
                    "document_id": document.metadata.get(
                        "document_id"
                    ),
                }
                for document in documents
            ]

            response = generate_answer(
                question,
                documents,
            )

            results.append({
                **base,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_sources": retrieved_sources,
                "response": response,
                "retrieved_count": len(documents),
                "error": None,
            })

        except Exception as exc:
            logger.exception(
                "Question %d failed",
                index,
            )

            results.append({
                **base,
                "retrieved_contexts": [],
                "retrieved_sources": [],
                "response": None,
                "retrieved_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return results


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write UTF-8 JSON output."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    testset = load_testset(DATASET_PATH)

    logger.info(
        "Loaded %d test cases from %s",
        len(testset),
        DATASET_PATH,
    )

    cases = run_pipeline(testset)

    # Save raw per-question pipeline outputs.
    write_json(
        RESULTS_DIR / "evaluation_results.json",
        cases,
    )

    csv_rows = []

    for case in cases:
        row = case.copy()

        for field in (
            "reference_contexts",
            "retrieved_contexts",
            "retrieved_sources",
        ):
            row[field] = json.dumps(
                row.get(field, []),
                ensure_ascii=False,
            )

        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        RESULTS_DIR / "evaluation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    successful_cases = [
        case for case in cases
        if case["error"] is None
    ]

    failed_count = len(cases) - len(successful_cases)

    # Run classical retrieval metrics.
    per_case, aggregate = evaluate_retrieval_cases(
        successful_cases,
        k_values=(1, 3, 5),
    )

    write_json(
        RESULTS_DIR / "classical_metrics.json",
        {
            "methodology": (
                "Heuristic normalized text containment. "
                "Not equivalent to manually labelled relevance."
            ),
            "per_case": per_case,
            "aggregate": aggregate,
        },
    )

    # Ragas is optional: a Ragas failure does not discard other reports.
    try:
        from evals.ragas_judge import evaluate_with_ragas

        ragas_report = evaluate_with_ragas(
            successful_cases
        )

    except Exception as exc:
        logger.exception(
            "Ragas evaluation did not complete."
        )

        ragas_report = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    write_json(
        RESULTS_DIR / "ragas_metrics.json",
        ragas_report,
    )

    summary = {
        "total_cases": len(cases),
        "successful_cases": len(successful_cases),
        "failed_cases": failed_count,
        "top_k": TOP_K,
        "classical_metrics": aggregate,
        "ragas_status": ragas_report.get(
            "status",
            "unknown",
        ),
    }

    write_json(
        RESULTS_DIR / "summary.json",
        summary,
    )

    print("\nEvaluation completed.")
    print(f"Total cases:      {len(cases)}")
    print(f"Successful cases: {len(successful_cases)}")
    print(f"Failed cases:     {failed_count}")
    print(
        "Ragas status:     "
        f"{ragas_report.get('status', 'unknown')}"
    )
    print(f"Reports directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
