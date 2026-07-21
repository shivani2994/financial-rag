"""Post-hoc metadata filtering, applied after hybrid retrieval and before
reranking, per the Module 4 flow: retrieve wide -> filter -> rerank.

BM25's contribution to the hybrid candidate set has no native metadata
filtering (unlike Chroma's `where` clause), so scoping happens here,
uniformly, on the combined candidate list instead.
"""

from langchain_core.documents import Document


def apply_metadata_filter(
    documents: list[Document],
    company: str | None = None,
    period: str | None = None,
) -> list[Document]:
    """Keep only documents matching the given company and/or period, in order."""
    filtered = documents
    if company is not None:
        filtered = [d for d in filtered if d.metadata.get("company") == company]
    if period is not None:
        filtered = [d for d in filtered if d.metadata.get("period") == period]
    return filtered
