"""
Builds and reloads the BM25 keyword index over the same chunks as Chroma.

BM25Retriever holds the full document list and its keyword statistics
in-memory (there's no incremental on-disk format like Chroma's), so it's
persisted here with a simple pickle so later modules -- and re-queries --
don't need to re-read chunks.jsonl and re-tokenize every document.
"""

import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config.settings import settings


def build_bm25_index(documents: list[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)

    settings.bm25_persist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_persist_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever


def load_bm25_index() -> BM25Retriever:
    with open(settings.bm25_persist_path, "rb") as f:
        return pickle.load(f)
