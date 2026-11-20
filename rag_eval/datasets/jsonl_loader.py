import json
from pathlib import Path

from rag_eval.core.schemas import BenchmarkCase


def load_jsonl(
    file_path: str | Path,
) -> list[BenchmarkCase]:

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}"
        )

    cases = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                data = json.loads(line)

                case = BenchmarkCase.model_validate(data)

                cases.append(case)

            except Exception as error:

                raise ValueError(
                    f"Invalid benchmark entry at "
                    f"line {line_number}: {error}"
                ) from error

    case_ids = [
        case.case_id for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Duplicate benchmark case IDs detected."
        )

    return cases