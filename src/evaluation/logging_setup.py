"""
Structured, per-query observability logging.

Appends one JSON object per query to `settings.query_log_path`: latency,
whether it was refused, the retrieval confidence score, and the top
retrieved sources -- logged even when refused, so a refusal is still
auditable (what almost-but-didn't clear the bar).

Also appends one run-summary line per full harness run to
`settings.eval_run_log_path` -- the aggregate RAGAS scores otherwise only
ever printed to stdout, plus the run's configuration, so a baseline isn't
lost the moment the terminal scrolls past it.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from src.generation.pipeline import Answer


def log_query(log_path: Path, answer: Answer, latency_seconds: float, top_n: int) -> None:
    top_sources = [
        {
            "company": doc.metadata.get("company"),
            "doc_type": doc.metadata.get("doc_type"),
            "period": doc.metadata.get("period"),
            "section_or_speaker": doc.metadata.get("section_or_speaker"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
        for doc in answer.retrieved_documents[:top_n]
    ]
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": answer.question,
        "latency_seconds": round(latency_seconds, 3),
        "refused": answer.refused,
        "refusal_reason": answer.reason,
        "top_retrieval_score": answer.top_retrieval_score,
        "top_retrieved_sources": top_sources,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _safe_mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


def log_run_summary(
    log_path: Path,
    ragas_result: object | None,
    refusal_accuracy: float | None,
    answered_count: int,
    refused_count: int,
    num_questions: int,
) -> None:
    """Append one summary line for a full evaluation harness run."""
    faithfulness = answer_relevancy = context_precision = None
    if ragas_result is not None:
        faithfulness = _safe_mean(ragas_result["faithfulness"])
        answer_relevancy = _safe_mean(ragas_result["answer_relevancy"])
        context_precision = _safe_mean(ragas_result["llm_context_precision_without_reference"])

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "refusal_accuracy": refusal_accuracy,
        "answered_count": answered_count,
        "refused_count": refused_count,
        "num_questions": num_questions,
        "llm_model": settings.ollama_model,
        "refusal_confidence_threshold": settings.refusal_confidence_threshold,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "ragas_judge_model": settings.ragas_judge_model,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
