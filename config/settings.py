"""
Central application configuration.

All settings are loaded from environment variables (or a `.env` file at the
project root) via pydantic-settings. Add a new field here whenever a later
module introduces a new tunable value, and mirror it in `.env.example` so the
list of required config stays honest.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, used to build default absolute paths for data/index directories.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"

    # --- Indexing (Module 3) ---
    processed_chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    chroma_collection_name: str = "financial_rag"
    bm25_persist_path: Path = PROJECT_ROOT / "data" / "bm25" / "bm25.pkl"

    # --- Embeddings & reranking (Module 3 / 4) ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval (Module 4) ---
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # --- Generation (Module 5) ---
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    refusal_confidence_threshold: float = 0.5

    # --- Serving (Module 6) ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Evaluation (Module 7) ---
    eval_question_set_path: Path = PROJECT_ROOT / "eval" / "question_set.xlsx"


settings = Settings()
