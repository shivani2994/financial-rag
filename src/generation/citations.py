"""
Citation enforcement.

The prompt asks the model to mark every claim with a passage number like
[2]. This module is what makes that a real guarantee rather than a request:
it parses which numbers actually appear in the answer and resolves each one
back to its source chunk's metadata -- a marker that doesn't correspond to
a real passage is silently dropped rather than trusted. If nothing resolves,
the caller (see pipeline.py) treats the answer as ungrounded and refuses.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    marker: int
    company: str | None
    doc_type: str | None
    period: str | None
    section_or_speaker: str | None
    source_path: str | None
    chunk_id: str | None


def extract_citation_markers(answer: str) -> list[int]:
    """Distinct passage numbers cited in the answer, in first-seen order."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def resolve_citations(answer: str, documents: list[Document]) -> list[Citation]:
    """Resolve cited passage numbers to the metadata of the chunk they name.

    `documents` must be in the same order they were numbered in the prompt
    (passage [1] is documents[0], etc). A cited number with no matching
    passage -- the model hallucinating a citation -- is dropped, not trusted.
    """
    citations = []
    for n in extract_citation_markers(answer):
        index = n - 1
        if 0 <= index < len(documents):
            meta = documents[index].metadata
            citations.append(
                Citation(
                    marker=n,
                    company=meta.get("company"),
                    doc_type=meta.get("doc_type"),
                    period=meta.get("period"),
                    section_or_speaker=meta.get("section_or_speaker"),
                    source_path=meta.get("source_path"),
                    chunk_id=meta.get("chunk_id"),
                )
            )
    return citations
