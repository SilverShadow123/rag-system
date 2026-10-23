
"""Classical retrieval metrics for RAG evaluation."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for basic string matching."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def context_is_relevant(
    retrieved_context: str,
    reference_contexts: Sequence[str],
) -> bool:
    """Heuristic relevance test based on normalized text containment."""
    retrieved = normalize_text(retrieved_context)

    if not retrieved:
        return False

    for reference in reference_contexts:
        reference = normalize_text(reference)

        # Ignore short passages to reduce accidental matches.
        if len(reference) < 40:
            continue

        if reference in retrieved or retrieved in reference:
            return True

    return False


def calculate_retrieval_metrics(
    retrieved_contexts: Sequence[str],
    reference_contexts: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Calculate heuristic retrieval metrics at specified cutoffs."""
    retrieved = list(retrieved_contexts)

    references = [
        ref
        for ref in reference_contexts
        if isinstance(ref, str) and len(normalize_text(ref)) >= 40
    ]

    relevance_by_rank = []

    for context in retrieved:
        matched = [
            index
            for index, reference in enumerate(references)
            if context_is_relevant(context, [reference])
        ]
        relevance_by_rank.append(matched)

    first_relevant_rank = next(
        (
            rank
            for rank, matches in enumerate(
                relevance_by_rank, start=1
            )
            if matches
        ),
        None,
    )

    results = {
        "retrieved_count": len(retrieved),
        "reference_context_count": len(references),
        "matching_method": "normalized text containment heuristic",
        "reciprocal_rank": (
            1.0 / first_relevant_rank
            if first_relevant_rank
            else 0.0
        ),
    }

    for k in k_values:
        if k <= 0:
            raise ValueError("Each k value must be positive.")

        top_matches = relevance_by_rank[:k]

        relevant_retrieved = sum(
            bool(matches) for matches in top_matches
        )

        matched_reference_indices = {
            reference_index
            for matches in top_matches
            for reference_index in matches
        }

        precision = (
            relevant_retrieved / len(top_matches)
            if top_matches
            else 0.0
        )

        recall = (
            len(matched_reference_indices) / len(references)
            if references
            else None
        )

        hit_rate = 1.0 if relevant_retrieved else 0.0

        reciprocal_rank = 0.0

        for rank, matches in enumerate(top_matches, start=1):
            if matches:
                reciprocal_rank = 1.0 / rank
                break

        results[f"precision_at_{k}"] = precision
        results[f"recall_at_{k}"] = recall
        results[f"hit_rate_at_{k}"] = hit_rate
        results[f"mrr_at_{k}"] = reciprocal_rank

    return results


def evaluate_retrieval_cases(
    cases: Sequence[dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate per-question and aggregate retrieval metrics."""
    per_case = []

    for case in cases:
        metrics = calculate_retrieval_metrics(
            retrieved_contexts=case.get(
                "retrieved_contexts", []
            ),
            reference_contexts=case.get(
                "reference_contexts", []
            ),
            k_values=k_values,
        )

        per_case.append({
            "user_input": case.get("user_input", ""),
            **metrics,
        })

    aggregate = {"case_count": len(per_case)}

    metric_keys = [
        key
        for key in (
            per_case[0].keys() if per_case else []
        )
        if key.startswith((
            "precision_at_",
            "recall_at_",
            "hit_rate_at_",
            "mrr_at_",
        ))
    ]

    for key in metric_keys:
        values = [
            row[key]
            for row in per_case
            if isinstance(row.get(key), (int, float))
        ]

        aggregate[key] = (
            sum(values) / len(values)
            if values
            else None
        )

    return per_case, aggregate
