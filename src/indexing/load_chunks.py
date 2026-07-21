"""Loads the metadata-rich chunks produced by ingestion (Module 2) as
LangChain Documents, ready to feed into both indexes in one pass.
"""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_documents(chunks_path: Path) -> list[Document]:
    """Read `chunks.jsonl` (one {chunk_id, text, metadata} object per line).

    `section_or_speaker` is dropped from a chunk's metadata when it's null
    (e.g. transcript front matter, memos) rather than stored as None --
    Chroma's metadata store only accepts str/int/float/bool values.
    """
    documents = []
    with open(chunks_path) as f:
        for line in f:
            chunk = json.loads(line)
            metadata = {"chunk_id": chunk["chunk_id"], **chunk["metadata"]}
            metadata = {k: v for k, v in metadata.items() if v is not None}
            documents.append(Document(page_content=chunk["text"], metadata=metadata))
    return documents
