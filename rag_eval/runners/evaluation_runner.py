from uuid import uuid4

from rag_eval.core.protocols import (
    RAGAdapter,
    Metric,
)

from rag_eval.core.registry import MetricRegistry

from rag_eval.core.schemas import (
    BenchmarkCase,
    EvaluationReport,
    MetricResult,
    QueryResult,
)


class EvaluationRunner:

    def __init__(
        self,
        adapter: RAGAdapter,
        metric_registry: MetricRegistry,
    ):

        self.adapter = adapter
        self.metric_registry = metric_registry

    def run(
        self,
        cases: list[BenchmarkCase],
        *,
        experiment_id: str,
        options: dict | None = None,
    ) -> EvaluationReport:

        if not cases:
            raise ValueError(
                "Cannot evaluate an empty benchmark."
            )

        case_ids = [
            case.case_id for case in cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Duplicate benchmark case IDs detected."
            )

        query_results = []

        # Phase 1: Execute all benchmark cases.
        for case in cases:

            request_id = str(uuid4())

            try:

                result = self.adapter.run(
                    case.question,
                    request_id=request_id,
                    options=options,
                )

                if not isinstance(result, QueryResult):
                    raise TypeError(
                        "Adapter must return a QueryResult."
                    )

                if result.case_id not in (None, case.case_id):
                    raise ValueError(
                        "Adapter returned a mismatched case_id."
                    )

                if result.request_id != request_id:
                    raise ValueError(
                        "Adapter returned a mismatched request_id."
                    )

                result = result.model_copy(
                    update={
                        "case_id": case.case_id,
                    }
                )

            except Exception as error:

                result = QueryResult(
                    request_id=request_id,
                    case_id=case.case_id,
                    answer=None,
                    retrieved_items=None,
                    status="error",
                    error=(
                        f"{type(error).__name__}: {error}"
                    ),
                    metadata={
                        "adapter": self.adapter.name,
                    },
                )

            query_results.append(result)

        successful_cases = sum(
            result.status == "success"
            for result in query_results
        )

        failed_cases = len(query_results) - successful_cases

        # Phase 2: Run registered evaluation metrics.
        metric_results = []

        for metric in self.metric_registry.get_all():

            try:

                scores = metric.evaluate(
                    cases,
                    query_results,
                )

                if not isinstance(scores, list):
                    raise TypeError(
                        f"Metric '{metric.name}' must return a list."
                    )

                for score in scores:

                    if not isinstance(score, MetricResult):
                        raise TypeError(
                            f"Metric '{metric.name}' returned "
                            "an invalid result."
                        )

                    if score.metric_name != metric.name:
                        raise ValueError(
                            "Metric result name does not match "
                            "registered metric."
                        )

                metric_results.extend(scores)

            except Exception as error:

                metric_results.append(
                    MetricResult(
                        metric_name=metric.name,
                        value=None,
                        status="error",
                        details={
                            "error": (
                                f"{type(error).__name__}: {error}"
                            )
                        },
                    )
                )

        # Phase 3: Determine experiment status.
        has_metric_errors = any(
            result.status == "error"
            for result in metric_results
        )

        if failed_cases == len(cases):
            status = "error"

        elif failed_cases > 0 or has_metric_errors:
            status = "partial"

        else:
            status = "success"

        return EvaluationReport(
            experiment_id=experiment_id,
            adapter_name=self.adapter.name,
            total_cases=len(cases),
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            query_results=query_results,
            metric_results=metric_results,
            status=status,
            metadata={
                "options": options or {},
                "registered_metrics": (
                    self.metric_registry.names()
                ),
            },
        )