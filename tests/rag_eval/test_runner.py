
import json

import pytest

from rag_eval.adapters.fake_adapter import FakeRAGAdapter
from rag_eval.core.registry import MetricRegistry
from rag_eval.core.schemas import MetricResult
from rag_eval.datasets.jsonl_loader import load_jsonl
from rag_eval.runners.evaluation_runner import EvaluationRunner


class AnswerPresenceMetric:
    """Checks whether the adapter returned a non-empty answer."""

    name = "answer_presence"

    def evaluate(self, cases, results):
        return [
            MetricResult(
                metric_name=self.name,
                value=1.0 if result.answer else 0.0,
                status="success",
                case_id=case.case_id,
            )
            for case, result in zip(cases, results)
        ]


class FailingMetric:
    name = "failing_metric"

    def evaluate(self, cases, results):
        raise RuntimeError("Simulated metric failure")


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "test_dataset.jsonl"

    content = "\n".join([
        json.dumps({
            "case_id": "q001",
            "question": "What accuracy did Model A achieve?",
            "expected_answer": "Model A achieved 92% accuracy.",
            "answerable": True,
            "expected_evidence": [],
            "tags": ["smoke-test", "table"],
        }),
        json.dumps({
            "case_id": "q002",
            "question": "What was yesterday's weather?",
            "expected_answer": "The document does not provide this information.",
            "answerable": False,
            "expected_evidence": [],
            "tags": ["smoke-test", "unanswerable"],
        }),
    ])

    path.write_text(content + "\n", encoding="utf-8")
    return path


@pytest.fixture
def benchmark_cases(dataset_path):
    return load_jsonl(dataset_path)


def test_metric_registry_registers_metric():
    registry = MetricRegistry()
    metric = AnswerPresenceMetric()

    registry.register(metric)

    assert registry.get("answer_presence") is metric
    assert "answer_presence" in registry.names()
    assert len(registry) == 1


def test_metric_registry_rejects_duplicate_metric():
    registry = MetricRegistry()

    registry.register(AnswerPresenceMetric())

    with pytest.raises(ValueError):
        registry.register(AnswerPresenceMetric())


def test_metric_registry_rejects_unknown_metric():
    registry = MetricRegistry()

    with pytest.raises(KeyError):
        registry.get("unknown_metric")


def test_dataset_loader_reads_jsonl(dataset_path):
    cases = load_jsonl(dataset_path)

    assert len(cases) == 2
    assert cases[0].case_id == "q001"
    assert cases[1].case_id == "q002"
    assert cases[0].answerable is True
    assert cases[1].answerable is False


def test_dataset_loader_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"

    entry = {
        "case_id": "q001",
        "question": "Question?",
        "expected_answer": "Answer",
        "answerable": True,
        "expected_evidence": [],
    }

    path.write_text(
        json.dumps(entry) + "\n" + json.dumps(entry) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl(path)


def test_runner_executes_fake_adapter(benchmark_cases):
    registry = MetricRegistry()
    registry.register(AnswerPresenceMetric())

    runner = EvaluationRunner(
        adapter=FakeRAGAdapter(),
        metric_registry=registry,
    )

    report = runner.run(
        cases=benchmark_cases,
        experiment_id="test-experiment",
    )

    assert report.total_cases == 2
    assert report.successful_cases == 2
    assert report.failed_cases == 0
    assert report.status == "success"
    assert len(report.query_results) == 2
    assert len(report.metric_results) == 2


def test_runner_preserves_case_ids(benchmark_cases):
    runner = EvaluationRunner(
        adapter=FakeRAGAdapter(),
        metric_registry=MetricRegistry(),
    )

    report = runner.run(
        cases=benchmark_cases,
        experiment_id="case-id-test",
    )

    result_case_ids = [
        result.case_id for result in report.query_results
    ]

    assert result_case_ids == ["q001", "q002"]


def test_runner_rejects_empty_cases():
    runner = EvaluationRunner(
        adapter=FakeRAGAdapter(),
        metric_registry=MetricRegistry(),
    )

    with pytest.raises(ValueError):
        runner.run(
            cases=[],
            experiment_id="empty-test",
        )


def test_runner_rejects_duplicate_case_ids(benchmark_cases):
    runner = EvaluationRunner(
        adapter=FakeRAGAdapter(),
        metric_registry=MetricRegistry(),
    )

    duplicate_cases = [benchmark_cases[0], benchmark_cases[0]]

    with pytest.raises(ValueError, match="Duplicate"):
        runner.run(
            cases=duplicate_cases,
            experiment_id="duplicate-test",
        )


def test_runner_handles_adapter_failure(benchmark_cases):
    class FailingAdapter:
        name = "failing_adapter"

        def run(self, question, request_id, options=None):
            raise RuntimeError("Simulated adapter failure")

    runner = EvaluationRunner(
        adapter=FailingAdapter(),
        metric_registry=MetricRegistry(),
    )

    report = runner.run(
        cases=benchmark_cases,
        experiment_id="adapter-failure-test",
    )

    assert report.total_cases == 2
    assert report.successful_cases == 0
    assert report.failed_cases == 2
    assert report.status == "error"
    assert all(
        result.status == "error"
        for result in report.query_results
    )


def test_runner_handles_metric_failure(benchmark_cases):
    registry = MetricRegistry()
    registry.register(FailingMetric())

    runner = EvaluationRunner(
        adapter=FakeRAGAdapter(),
        metric_registry=registry,
    )

    report = runner.run(
        cases=benchmark_cases,
        experiment_id="metric-failure-test",
    )

    assert report.total_cases == 2
    assert report.successful_cases == 2
    assert report.failed_cases == 0
    assert report.status == "partial"
    assert len(report.metric_results) == 1
    assert report.metric_results[0].status == "error"
