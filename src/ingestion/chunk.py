"""A single chunk of ingested text plus the metadata it carries."""

from pydantic import BaseModel

from src.common.metadata import ChunkMetadata


class Chunk(BaseModel):
    """One chunk produced by ingestion, ready for indexing in Module 3."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
