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

    # --- Ingestion (Module 2) ---
    processed_chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    chunk_max_chars: int = 1500

    # --- Indexing (Module 3) ---
    chroma_collection_name: str = "financial_rag"
    bm25_persist_path: Path = PROJECT_ROOT / "data" / "bm25" / "bm25.pkl"

    # --- Embeddings & reranking (Module 3 / 4) ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval (Module 4) ---
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # --- Generation (Module 5) ---
    ollama_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://localhost:11434"
    # bge-reranker-base's sigmoid score on this corpus: confident in-corpus hits
    # land at ~0.4-1.0, clearly out-of-corpus queries mostly land under ~0.06.
    # 0.1 sits in between; see Module 5 review notes for the calibration data
    # and its known edge cases. Retune this once Module 7's refusal question
    # set gives a real accuracy number to optimize against.
    refusal_confidence_threshold: float = 0.1

    # --- Refusal gate hardening: deterministic pre-generation checks ---
    # Ticker -> alternate names/spellings that count as naming that company
    # in a question. Tickers themselves (KO, PEP, MDLZ) are always matched
    # too; these are the additional aliases the scope-coverage check
    # recognizes. Keyword rule set, not a model -- deterministic and
    # identical on every run.
    company_aliases: dict[str, list[str]] = {
        "KO": ["coca-cola", "coca cola", "the coca-cola company"],
        "PEP": ["pepsico", "pepsi"],
        "MDLZ": ["mondelez", "mondelez international", "mondelēz", "mondelēz international"],
    }
    # Phrases that mark a question as asking for live/current market data
    # (a stock's price right now, today's valuation) -- evidence this corpus
    # of static filings and transcripts can never contain, regardless of how
    # confident retrieval is. Matched case-insensitively as substrings.
    live_market_data_keywords: list[str] = [
        "stock price",
        "share price",
        "trading price",
        "current price",
        "current valuation",
        "today's price",
        "today's valuation",
        "market cap",
        "market capitalization",
        "market capitalisation",
        "trading at",
        "stock quote",
        "real-time price",
        "live price",
    ]

    # --- Serving (Module 6) ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Evaluation (Module 7) ---
    eval_question_set_path: Path = PROJECT_ROOT / "eval" / "question_set.xlsx"
    query_log_path: Path = PROJECT_ROOT / "logs" / "queries.jsonl"
    eval_run_log_path: Path = PROJECT_ROOT / "logs" / "eval_runs.jsonl"
    top_sources_logged: int = 3
    # Deliberately separate from `ollama_model` (the generation model under
    # test) and held fixed across comparison runs: if the judge changed
    # along with the generation model, a RAGAS score difference could be
    # the judge behaving differently rather than generation improving.
    ragas_judge_model: str = "llama3.2:1b"

    # --- Fast evaluation gate (CI): deterministic, no LLM judge ---
    gate_thresholds_path: Path = PROJECT_ROOT / "eval" / "gate_thresholds.json"
    fast_gate_result_path: Path = PROJECT_ROOT / "eval" / "fast_gate_result.json"


settings = Settings()
