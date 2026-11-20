
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRef(ContractModel):
    """
    Architecture-independent reference to original source evidence.
    """

    source_id: str = Field(min_length=1)
    source_name: str | None = None

    page_number: int | None = Field(default=None, ge=1)

    element_id: str | None = None

    element_type: Literal[
        "text",
        "table",
        "image",
        "image_description",
        "other",
    ] = "other"

    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_character_range(self):
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError(
                "end_char cannot be smaller than start_char."
            )

        return self


class BenchmarkCase(ContractModel):
    """
    One independently evaluated question.
    """

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    expected_answer: str | None = None
    answerable: bool

    expected_evidence: list[EvidenceRef] = Field(
        default_factory=list
    )

    question_type: str | None = None

    modality: Literal[
        "text",
        "table",
        "image",
        "mixed",
        "unknown",
    ] = "unknown"

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentElement(ContractModel):
    """
    Normalized representation of an extracted source element.
    """

    element_id: str = Field(min_length=1)

    element_type: Literal[
        "text",
        "table",
        "image",
        "image_description",
        "other",
    ]

    content: str | None = None

    evidence: list[EvidenceRef] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkOutput(ContractModel):
    """
    Architecture-independent representation of an indexed chunk.
    """

    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)

    evidence: list[EvidenceRef] = Field(
        default_factory=list
    )

    token_count: int | None = Field(default=None, ge=0)

    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(ContractModel):
    document_id: str = Field(min_length=1)

    status: Literal["success", "partial", "error"]

    extracted_elements: list[ContentElement] = Field(
        default_factory=list
    )

    generated_chunks: list[ChunkOutput] = Field(
        default_factory=list
    )

    statistics: dict[str, Any] = Field(
        default_factory=dict
    )

    timings_ms: dict[str, float] = Field(
        default_factory=dict
    )

    errors: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedItem(ContractModel):
    """
    Normalized representation of one ranked retrieval result.
    """

    item_id: str = Field(min_length=1)

    content: str

    rank: int = Field(ge=1)

    score: float | None = None

    evidence: list[EvidenceRef] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(ContractModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    model: str | None = None
    provider: str | None = None

    cost_usd: float | None = Field(default=None, ge=0)


class RetrievalResult(ContractModel):
    question: str

    items: list[RetrievedItem] = Field(
        default_factory=list
    )

    duration_ms: float = Field(ge=0)

    status: Literal["success", "error"]

    error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(ContractModel):
    answer: str | None = None

    duration_ms: float = Field(ge=0)

    token_usage: TokenUsage | None = None

    status: Literal["success", "error"]

    error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(ContractModel):
    """
    Normalized end-to-end response from any RAG system.

    retrieved_items=None means retrieval information is unavailable.
    retrieved_items=[] means retrieval ran but returned no results.
    """

    request_id: str = Field(min_length=1)

    answer: str | None = None

    retrieved_items: list[RetrievedItem] | None = None

    timings_ms: dict[str, float] = Field(
        default_factory=dict
    )

    token_usage: TokenUsage | None = None

    status: Literal[
        "success",
        "partial",
        "error",
    ]

    error: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(ContractModel):
    """
    Standard output from any evaluation metric.
    """

    metric_name: str = Field(min_length=1)

    value: float | None = None

    status: Literal[
        "success",
        "skipped",
        "error",
    ] = "success"

    case_id: str | None = None

    unit: str | None = None

    details: dict[str, Any] = Field(
        default_factory=dict
    )
