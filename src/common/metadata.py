"""
Shared chunk metadata schema.

Every chunk produced by ingestion (Module 2) and stored in the vector index
(Module 3) carries this metadata. Keeping the schema in one place lets
retrieval filtering (Module 4) and citation (Module 5) rely on a single
source of truth for field names and allowed values.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    TRANSCRIPT = "transcript"
    MEMO = "memo"


class Company(str, Enum):
    KO = "KO"
    PEP = "PEP"
    MDLZ = "MDLZ"


class ChunkMetadata(BaseModel):
    """Metadata attached to a single chunk of a source document."""

    company: Company
    doc_type: DocType
    period: str = Field(
        description="Reporting period as it appears in the filename, e.g. 'FY2025' or '2026Q1'."
    )
    section_or_speaker: str | None = Field(
        default=None,
        description=(
            "Filing item (e.g. 'Item 7 - MD&A') for 10-K/10-Q chunks, or speaker "
            "name for transcript chunks. None when not applicable (e.g. memos)."
        ),
    )
    source_path: str = Field(description="Path to the source PDF this chunk was extracted from.")
