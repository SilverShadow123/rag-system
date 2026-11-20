from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from collections.abc import Mapping, Sequence

from rag_eval.core.schemas import (
    GenerationResult,
    IngestionResult,
    MetricResult,
    QueryResult,
    RetrievalResult,
    RetrievedItem,
    BenchmarkCase,
)


Options = Mapping[str, Any] | None


@runtime_checkable
class IngestionAdapter(Protocol):

    name: str

    def ingest(
        self,
        source: str | Path,
        *,
        options: Options = None,
    ) -> IngestionResult:
        ...


@runtime_checkable
class RetrievalAdapter(Protocol):

    name: str

    def retrieve(
        self,
        question: str,
        *,
        request_id: str,
        top_k: int = 5,
        options: Options = None,
    ) -> RetrievalResult:
        ...


@runtime_checkable
class GenerationAdapter(Protocol):

    name: str

    def generate(
        self,
        question: str,
        retrieved_items: Sequence[RetrievedItem],
        *,
        request_id: str,
        options: Options = None,
    ) -> GenerationResult:
        ...


@runtime_checkable
class RAGAdapter(Protocol):

    name: str

    def run(
        self,
        question: str,
        *,
        request_id: str,
        options: Options = None,
    ) -> QueryResult:
        ...


@runtime_checkable
class Metric(Protocol):

    name: str

    def evaluate(
        self,
        cases: Sequence[BenchmarkCase],
        results: Sequence[QueryResult],
    ) -> list[MetricResult]:
        ...