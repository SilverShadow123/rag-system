from rag_eval.core.protocols import Metric


class MetricRegistry:

    def __init__(self):
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:

        if not metric.name:
            raise ValueError(
                "Metric name cannot be empty."
            )

        if metric.name in self._metrics:
            raise ValueError(
                f"Metric already registered: {metric.name}"
            )

        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:

        try:
            return self._metrics[name]

        except KeyError:
            raise KeyError(
                f"Metric not registered: {name}"
            )

    def get_all(self) -> list[Metric]:

        return list(self._metrics.values())

    def names(self) -> list[str]:

        return list(self._metrics.keys())

    def __len__(self) -> int:

        return len(self._metrics)