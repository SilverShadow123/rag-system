
import pytest
from pydantic import ValidationError

from rag_eval.core.protocols import (
    RAGAdapter,
    RetrievalAdapter,
)

from rag_eval.core.schemas import (
    BenchmarkCase,
    EvidenceRef,
    MetricResult,
    QueryResult,
    RetrievedItem,
)

from rag_eval.adapters.fake_adapter import FakeRAGAdapter


def test_benchmark_case_creation():

    case = BenchmarkCase(
        case_id="q001",
        question="What accuracy did Model A achieve?",
        expected_answer="Model A achieved 92% accuracy.",
        answerable=True,
        expected_evidence=[
            EvidenceRef(
                source_id="research-paper-001",
                page_number=9,
                element_type="table",
            )
        ],
    )

    assert case.case_id == "q001"
    assert len(case.expected_evidence) == 1


def test_evidence_rejects_invalid_page_number():

    with pytest.raises(ValidationError):

        EvidenceRef(
            source_id="research-paper-001",
            page_number=0,
        )


def test_evidence_rejects_invalid_character_range():

    with pytest.raises(ValidationError):

        EvidenceRef(
            source_id="research-paper-001",
            start_char=100,
            end_char=50,
        )


def test_retrieved_item_requires_positive_rank():

    with pytest.raises(ValidationError):

        RetrievedItem(
            item_id="chunk-001",
            content="Some content.",
            rank=0,
        )


def test_query_result_distinguishes_unavailable_retrieval():

    result = QueryResult(
        request_id="request-001",
        answer="Some answer.",
        retrieved_items=None,
        status="success",
    )

    assert result.retrieved_items is None


def test_query_result_distinguishes_empty_retrieval():

    result = QueryResult(
        request_id="request-002",
        answer=None,
        retrieved_items=[],
        status="success",
    )

    assert result.retrieved_items == []


def test_metric_result_creation():

    result = MetricResult(
        case_id="q001",
        metric_name="faithfulness",
        value=0.91,
        unit="score",
    )

    assert result.metric_name == "faithfulness"
    assert result.value == 0.91


def test_fake_adapter_implements_interfaces():

    adapter = FakeRAGAdapter()

    assert isinstance(adapter, RAGAdapter)
    assert isinstance(adapter, RetrievalAdapter)


def test_fake_adapter_returns_normalized_result():

    adapter = FakeRAGAdapter()

    result = adapter.run(
        "What accuracy did Model A achieve?",
        request_id="request-001",
    )

    assert isinstance(result, QueryResult)

    assert result.status == "success"

    assert "92%" in result.answer

    assert len(result.retrieved_items) == 1

    evidence = result.retrieved_items[0].evidence[0]

    assert evidence.source_name == "research.pdf"
    assert evidence.page_number == 9


def test_fake_adapter_handles_unanswerable_question():

    adapter = FakeRAGAdapter()

    result = adapter.run(
        "What was the weather yesterday?",
        request_id="request-002",
    )

    assert result.status == "success"

    assert result.retrieved_items == []

    assert (
        "not contain enough evidence"
        in result.answer
    )


def test_adapter_can_limit_retrieval_results():

    adapter = FakeRAGAdapter()

    result = adapter.run(
        "What accuracy did Model A achieve?",
        request_id="request-003",
        options={"top_k": 1},
    )

    assert len(result.retrieved_items) == 1
