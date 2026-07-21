"""Request/response models for the serving API."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, description="The analyst's plain-language question.")
    company: str | None = Field(
        default=None, description="Optional ticker to scope retrieval, e.g. 'KO', 'PEP', 'MDLZ'."
    )
    period: str | None = Field(
        default=None, description="Optional reporting period to scope retrieval, e.g. '2026Q1', 'FY2025'."
    )


class CitationResponse(BaseModel):
    marker: int
    company: str | None
    doc_type: str | None
    period: str | None
    section_or_speaker: str | None
    source_path: str | None
    chunk_id: str | None


class AnswerResponse(BaseModel):
    question: str
    refused: bool
    reason: str | None = Field(
        default=None, description="Why the system refused, present only when refused=true."
    )
    answer: str | None = Field(
        default=None, description="The grounded answer, present only when refused=false."
    )
    citations: list[CitationResponse] = Field(
        default_factory=list, description="Source chunks backing the answer, empty when refused."
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Top reranked retrieval score for this question -- the same "
            "signal the refusal gate is thresholded on."
        ),
    )
