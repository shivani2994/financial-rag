"""
Hybrid retrieval: dense (Chroma) + keyword (BM25), combined by LangChain's
EnsembleRetriever via reciprocal rank fusion.

Loads both indexes exactly as Module 3 persisted them -- no rebuilding, no
re-embedding. `id_key="chunk_id"` tells the ensemble how to recognize the
same chunk surfaced by both retrievers, so it's fused into one ranked
result instead of counted twice.
"""

from langchain_classic.retrievers import EnsembleRetriever

from config.settings import settings
from src.indexing.bm25_index import load_bm25_index
from src.indexing.vector_store import load_vector_store


def build_hybrid_retriever(k: int) -> EnsembleRetriever:
    vector_store = load_vector_store()
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # load_bm25_index() is cached and shared across every call (see its
    # docstring) -- copy rather than mutate its `.k` in place, so concurrent
    # requests with different candidate widths can't race on shared state.
    bm25_retriever = load_bm25_index().model_copy(update={"k": k})

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        id_key="chunk_id",
    )
