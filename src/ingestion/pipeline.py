"""
Ingestion pipeline: turns the PDFs in `data/` into metadata-rich chunks.

Run with:

    uv run python -m src.ingestion.pipeline

Each run rebuilds the full chunk list from scratch and overwrites the output
file at `settings.processed_chunks_path` (JSON Lines, one chunk per line).
Rebuilding from scratch on every run -- rather than appending -- is what
makes ingestion reproducible: re-running never duplicates a chunk, because
there's nothing to duplicate into. Chunk IDs are also deterministic (a hash
of source path + position), so the same input always produces the same IDs,
which Module 3's indexing step can rely on to upsert cleanly.
"""

import hashlib
import json
from pathlib import Path

from config.settings import settings
from src.common.metadata import ChunkMetadata
from src.ingestion.chunk import Chunk
from src.ingestion.chunker import chunk_document
from src.ingestion.metadata_parser import parse_filename_metadata
from src.ingestion.pdf_loader import load_pdf_text


def _make_chunk_id(source_path: str, index: int) -> str:
    digest = hashlib.sha256(f"{source_path}::{index}".encode()).hexdigest()
    return digest[:16]


def discover_pdfs(data_dir: Path) -> list[Path]:
    """Find every source PDF under data/<TICKER>/, in a stable sorted order."""
    return sorted(data_dir.glob("*/*.pdf"))


def ingest_document(pdf_path: Path) -> list[Chunk]:
    """Load, tag, and chunk a single PDF."""
    doc_metadata = parse_filename_metadata(pdf_path)
    text = load_pdf_text(pdf_path)
    pieces = chunk_document(text, doc_metadata.doc_type, settings.chunk_max_chars)

    chunks = []
    for index, (chunk_text, section_or_speaker) in enumerate(pieces):
        metadata = ChunkMetadata(
            company=doc_metadata.company,
            doc_type=doc_metadata.doc_type,
            period=doc_metadata.period,
            section_or_speaker=section_or_speaker,
            source_path=doc_metadata.source_path,
        )
        chunk_id = _make_chunk_id(doc_metadata.source_path, index)
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
    return chunks


def run_ingestion() -> list[Chunk]:
    pdf_paths = discover_pdfs(settings.data_dir)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {settings.data_dir}/<TICKER>/*.pdf")

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        doc_chunks = ingest_document(pdf_path)
        all_chunks.extend(doc_chunks)
        print(f"{pdf_path}: {len(doc_chunks)} chunks")

    chunk_ids = [c.chunk_id for c in all_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Duplicate chunk_id detected within a single ingestion run.")

    settings.processed_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_chunks_path, "w") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_paths)} documents")
    print(f"Written to {settings.processed_chunks_path}")
    return all_chunks


def _print_samples(chunks: list[Chunk], n: int = 3) -> None:
    print(f"\n--- {n} sample chunks ---")
    for chunk in chunks[:: max(1, len(chunks) // n)][:n]:
        print(f"\nchunk_id: {chunk.chunk_id}")
        print(f"metadata: {chunk.metadata.model_dump()}")
        preview = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
        print(f"text: {preview}")


if __name__ == "__main__":
    chunks = run_ingestion()
    _print_samples(chunks)
