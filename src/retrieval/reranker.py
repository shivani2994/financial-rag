"""
Second-stage reranking with bge-reranker-base, a cross-encoder that scores
each (query, candidate) pair jointly -- more precise than the bi-encoder
similarity scores from hybrid retrieval, but too slow to run over the whole
corpus, hence it only re-scores the already-narrowed candidate set.
"""

from sentence_transformers import CrossEncoder

from config.settings import settings
from langchain_core.documents import Document

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def rerank(
    query: str, documents: list[Document], top_k: int
) -> list[tuple[Document, float]]:
    """Score each document against the query and return the top_k, best first."""
    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = get_reranker().predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
