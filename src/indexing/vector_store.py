"""
Builds and reloads the persistent Chroma vector store.

`build_vector_store` always rebuilds from scratch: it wipes any existing
persisted collection first, then re-embeds and re-adds every document with
its deterministic `chunk_id` (carried over from ingestion) as its Chroma id.
That mirrors ingestion's own "rebuild from scratch each run" approach to
reproducibility, so re-running indexing can never leave duplicate or stale
vectors behind.

`load_vector_store` does the opposite: it opens the already-persisted
collection on disk without recomputing any embeddings, for querying.
"""

import shutil
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.indexing.embeddings import get_embedding_function


def build_vector_store(documents: list[Document]) -> Chroma:
    if settings.chroma_persist_dir.exists():
        shutil.rmtree(settings.chroma_persist_dir)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    ids = [doc.metadata["chunk_id"] for doc in documents]
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_function(),
        ids=ids,
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_dir),
    )


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Open the existing persisted collection without rebuilding it.

    Cached: Module 6 serves many requests from one long-lived process, and
    Chroma reads its on-disk index (and reloads the embedding model) each
    time it's constructed -- pointless to redo per request when the
    collection on disk hasn't changed since the process started.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(settings.chroma_persist_dir),
    )
