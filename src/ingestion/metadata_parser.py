"""
Parses document-level metadata (company, doc_type, period) out of a source
PDF's filename and folder, per the convention in CLAUDE.md Section 9:

    data/<TICKER>/<TICKER>_<DOCTYPE>_<PERIOD>.pdf

e.g. data/KO/KO_10-K_FY2025.pdf, data/KO/KO_transcript_2026Q1.pdf.

Chunk-level metadata (section_or_speaker) is NOT parsed here -- it varies
within a single document and is filled in later by the chunker.
"""

from dataclasses import dataclass
from pathlib import Path

from src.common.metadata import Company, DocType


@dataclass
class DocumentMetadata:
    """Document-level metadata, shared by every chunk from one source PDF."""

    company: Company
    doc_type: DocType
    period: str
    source_path: str


def parse_filename_metadata(pdf_path: Path) -> DocumentMetadata:
    """Parse company, doc_type, and period from a PDF's filename and folder.

    Raises ValueError with a descriptive message if the filename doesn't
    follow the `<TICKER>_<DOCTYPE>_<PERIOD>.pdf` convention, or if the
    filename's ticker doesn't match the folder it lives in -- both are
    treated as hard errors rather than silently mis-tagged data, since a
    corpus this small should never need a guess.
    """
    parts = pdf_path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"{pdf_path}: filename '{pdf_path.name}' does not match the "
            "'<TICKER>_<DOCTYPE>_<PERIOD>.pdf' convention (expected exactly "
            "3 underscore-separated parts)."
        )
    company_token, doc_type_token, period_token = parts

    folder_token = pdf_path.parent.name
    if company_token != folder_token:
        raise ValueError(
            f"{pdf_path}: filename ticker '{company_token}' does not match "
            f"its parent folder 'data/{folder_token}/'. Fix the filename or "
            "move it to the matching folder."
        )

    try:
        company = Company(company_token)
    except ValueError as exc:
        raise ValueError(
            f"{pdf_path}: '{company_token}' is not a known company "
            f"({[c.value for c in Company]})."
        ) from exc

    try:
        doc_type = DocType(doc_type_token)
    except ValueError as exc:
        raise ValueError(
            f"{pdf_path}: '{doc_type_token}' is not a known doc_type "
            f"({[d.value for d in DocType]})."
        ) from exc

    return DocumentMetadata(
        company=company,
        doc_type=doc_type,
        period=period_token,
        source_path=str(pdf_path),
    )
