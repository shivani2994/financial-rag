"""
Builds and reloads the BM25 keyword index over the same chunks as Chroma.

BM25Retriever holds the full document list and its keyword statistics
in-memory (there's no incremental on-disk format like Chroma's), so it's
persisted here with a simple pickle so later modules -- and re-queries --
don't need to re-read chunks.jsonl and re-tokenize every document.
"""

import pickle
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config.settings import settings


def build_bm25_index(documents: list[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)

    settings.bm25_persist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_persist_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever


@lru_cache(maxsize=1)
def load_bm25_index() -> BM25Retriever:
    """Cached: Module 6 serves many requests from one long-lived process,
    and re-unpickling the same file on every request buys nothing once it's
    already in memory. Callers must not mutate the returned retriever in
    place (e.g. its `.k`) since it's shared across every caller -- see
    `build_hybrid_retriever`, which copies it instead.
    """
    with open(settings.bm25_persist_path, "rb") as f:
        return pickle.load(f)
