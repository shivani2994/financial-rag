"""
Retrieval pipeline: hybrid retrieve -> filter -> rerank.

Run with:

    uv run python -m src.retrieval.pipeline "your question" [--company KO] [--period 2026Q1]

Three explicit stages, matching the Module 4 spec:

1. Hybrid retrieval (dense + BM25, via Module 3's persisted indexes) pulls a
   wide candidate pool -- wider still when a filter is requested, so there's
   enough left over after filtering to rerank meaningfully.
2. An optional metadata filter (company and/or period) scopes that pool.
3. The bge-reranker-base cross-encoder re-scores what's left for final
   precision, keeping the top `rerank_top_k`.

Every stage returns LangChain Documents, whose `.metadata` carries the full
chunk metadata (company, doc_type, period, section_or_speaker, source_path,
chunk_id) through to the final result, so every result's source is known.
"""

import argparse
from dataclasses import dataclass

from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.filters import apply_metadata_filter
from src.retrieval.hybrid import build_hybrid_retriever
from src.retrieval.reranker import rerank

# When a filter is requested, retrieve a wider candidate pool up front so
# filtering doesn't starve the reranker of anything left to work with.
_FILTERED_CANDIDATE_MULTIPLIER = 3
_MAX_CANDIDATE_K = 30


@dataclass
class RetrievalResult:
    query: str
    hybrid_candidates: list[Document]  # stage 1 output, pre-filter
    filtered_candidates: list[Document]  # stage 2 output, pre-rerank
    reranked: list[tuple[Document, float]]  # stage 3 output, final


def retrieve(
    query: str,
    company: str | None = None,
    period: str | None = None,
    final_k: int | None = None,
) -> RetrievalResult:
    final_k = final_k or settings.rerank_top_k

    candidate_k = settings.retrieval_top_k
    if company is not None or period is not None:
        candidate_k = min(candidate_k * _FILTERED_CANDIDATE_MULTIPLIER, _MAX_CANDIDATE_K)

    hybrid_retriever = build_hybrid_retriever(candidate_k)
    hybrid_candidates = hybrid_retriever.invoke(query)

    filtered_candidates = apply_metadata_filter(hybrid_candidates, company, period)

    reranked = rerank(query, filtered_candidates, final_k)

    return RetrievalResult(
        query=query,
        hybrid_candidates=hybrid_candidates,
        filtered_candidates=filtered_candidates,
        reranked=reranked,
    )


def _format_doc(doc: Document, score: float | None = None) -> str:
    meta = doc.metadata
    header = f"[{meta.get('company')} {meta.get('doc_type')} {meta.get('period')} | {meta.get('section_or_speaker', '(none)')}]"
    if score is not None:
        header = f"{header} score={score:.4f}"
    preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
    return f"{header}\n{preview}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Module 4 retrieval pipeline.")
    parser.add_argument("query")
    parser.add_argument("--company", default=None)
    parser.add_argument("--period", default=None)
    args = parser.parse_args()

    result = retrieve(args.query, company=args.company, period=args.period)

    print(f"Query: {result.query!r}  (company={args.company}, period={args.period})")
    print(f"\n--- stage 1: hybrid candidates ({len(result.hybrid_candidates)}) ---")
    for doc in result.hybrid_candidates:
        print(_format_doc(doc))
        print()

    print(f"--- stage 2: after metadata filter ({len(result.filtered_candidates)}) ---")
    for doc in result.filtered_candidates:
        print(_format_doc(doc))
        print()

    print(f"--- stage 3: after rerank (top {len(result.reranked)}) ---")
    for doc, score in result.reranked:
        print(_format_doc(doc, score))
        print()
