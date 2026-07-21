"""
Indexing pipeline: builds both search indexes from ingestion's chunks.

Run with:

    uv run python -m src.indexing.pipeline

Loads `chunks.jsonl` once, then builds the Chroma vector store (bge
embeddings) and the BM25 keyword index from the same document list in one
pass, persisting both to disk.
"""

from config.settings import settings
from src.indexing.bm25_index import build_bm25_index
from src.indexing.load_chunks import load_documents
from src.indexing.vector_store import build_vector_store


def run_indexing() -> None:
    documents = load_documents(settings.processed_chunks_path)
    print(f"Loaded {len(documents)} chunks from {settings.processed_chunks_path}")

    vector_store = build_vector_store(documents)
    print(
        f"Chroma: {vector_store._collection.count()} vectors persisted to "
        f"{settings.chroma_persist_dir}"
    )

    build_bm25_index(documents)
    print(f"BM25: {len(documents)} documents persisted to {settings.bm25_persist_path}")


def _sample_query(query: str, k: int = 3) -> None:
    """Reload the persisted Chroma store (no rebuild) and run a similarity query."""
    from src.indexing.vector_store import load_vector_store

    vector_store = load_vector_store()
    print(f"\n--- sample similarity query: {query!r} ---")
    for doc in vector_store.similarity_search(query, k=k):
        meta = doc.metadata
        print(
            f"\n[{meta.get('company')} {meta.get('doc_type')} {meta.get('period')} "
            f"| {meta.get('section_or_speaker', '(none)')}]"
        )
        print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))


if __name__ == "__main__":
    run_indexing()
    _sample_query("What drove revenue growth this quarter?")
