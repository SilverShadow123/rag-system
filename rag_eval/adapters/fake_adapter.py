
from collections.abc import Mapping
from typing import Any

from rag_eval.core.schemas import (
    EvidenceRef,
    QueryResult,
    RetrievedItem,
    RetrievalResult,
)


class FakeRAGAdapter:

    name = "fake-rag"

    def retrieve(
        self,
        question: str,
        *,
        request_id: str,
        top_k: int = 5,
        options: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:

        normalized_question = question.casefold()

        items = []

        if (
            "model a" in normalized_question
            and "accuracy" in normalized_question
        ):
            items.append(
                RetrievedItem(
                    item_id="fake-chunk-001",
                    content=(
                        "Model A achieved 92% accuracy "
                        "in the experimental evaluation."
                    ),
                    rank=1,
                    score=0.99,
                    evidence=[
                        EvidenceRef(
                            source_id="research-paper-001",
                            source_name="research.pdf",
                            page_number=9,
                            element_id="table-002",
                            element_type="table",
                        )
                    ],
                )
            )

        return RetrievalResult(
            question=question,
            items=items[:top_k],
            duration_ms=0.0,
            status="success",
            metadata={
                "request_id": request_id,
                "adapter": self.name,
                "synthetic": True,
            },
        )

    def run(
        self,
        question: str,
        *,
        request_id: str,
        options: Mapping[str, Any] | None = None,
    ) -> QueryResult:

        retrieval = self.retrieve(
            question,
            request_id=request_id,
            top_k=int((options or {}).get("top_k", 5)),
        )

        if retrieval.items:
            answer = (
                "Model A achieved 92% accuracy "
                "in the experimental evaluation."
            )
        else:
            answer = (
                "The demo corpus does not contain enough "
                "evidence to answer this question."
            )

        return QueryResult(
            request_id=request_id,
            answer=answer,
            retrieved_items=retrieval.items,
            timings_ms={
                "retrieval": 0.0,
                "generation": 0.0,
                "total": 0.0,
            },
            status="success",
            metadata={
                "adapter": self.name,
                "synthetic": True,
            },
        )
