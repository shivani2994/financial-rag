"""
Deterministic scope-coverage check, with a bounded repair step.

Parses the question for named companies (tickers and known aliases) and
named periods (e.g. "Q1 2026", "2026Q1", "FY2025"), then compares that
scope against the company/period metadata actually carried on the reranked
chunks handed up from retrieval. A missing (company, period) pair is a real
gap regardless of how confident the reranker's scores looked on the
original set -- the corpus may hold the answer, but the evidence in hand
doesn't.

Rather than refuse the moment a gap is found, `attempt_scope_repair` first
tries to fill it: for each missing pair, it runs one targeted retrieval
scoped to that exact company/period (reusing `retrieve`'s own metadata
filter -- no new retrieval logic), merges the results into the original
set, and reranks the merge with the existing bge-reranker-base cross
encoder (reusing `rerank` directly). The gap is checked exactly once more
against that merged, reranked set. If it's closed, generation proceeds
with the merged evidence; if not, it refuses, naming what's still missing
and noting that a targeted retrieval was already attempted. The repair
never loops -- one attempt, then a final decision either way.

Gap-finding itself is regex and set comparisons only: no LLM call, no
randomness, identical output for the same question and documents on every
run. The repair step's retrieval and reranking are exactly as deterministic
as they already were elsewhere in the pipeline -- nothing new is
introduced that wasn't already used for the initial retrieval.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from config.settings import settings
from src.retrieval.pipeline import retrieve
from src.retrieval.reranker import rerank

# "Q1 2026", "Q1, 2026", "Q1-2026" -> ("1", "2026")
_QUARTER_THEN_YEAR_RE = re.compile(r"\bQ([1-4])[\s,-]+(\d{4})\b", re.IGNORECASE)
# "2026Q1", "2026 Q1" (metadata's own period format, in case a question is
# phrased that way directly) -> ("2026", "1")
_YEAR_THEN_QUARTER_RE = re.compile(r"\b(\d{4})[\s-]?Q([1-4])\b", re.IGNORECASE)
# "FY2025", "FY 2025", "FY-2025"
_FISCAL_YEAR_RE = re.compile(r"\bFY[\s-]?(\d{4})\b", re.IGNORECASE)


def extract_mentioned_companies(question: str, company_aliases: dict[str, list[str]]) -> set[str]:
    """Tickers named directly (KO, PEP, MDLZ) or via a configured alias."""
    mentioned = set()
    question_lower = question.lower()
    for ticker, aliases in company_aliases.items():
        if re.search(rf"\b{re.escape(ticker)}\b", question):
            mentioned.add(ticker)
            continue
        if any(alias.lower() in question_lower for alias in aliases):
            mentioned.add(ticker)
    return mentioned


def extract_mentioned_periods(question: str) -> set[str]:
    """Periods named in the question, normalized to the corpus's own
    metadata format (e.g. "2026Q1", "FY2025").
    """
    periods = set()
    for quarter, year in _QUARTER_THEN_YEAR_RE.findall(question):
        periods.add(f"{year}Q{quarter}")
    for year, quarter in _YEAR_THEN_QUARTER_RE.findall(question):
        periods.add(f"{year}Q{quarter}")
    for year in _FISCAL_YEAR_RE.findall(question):
        periods.add(f"FY{year}")
    return periods


# A named (company, period) combination the question cares about. Either
# field may be None when the question only names one dimension (e.g. a
# company with no specific period) -- None means "unconstrained" on that
# dimension when checking coverage, not "must be absent".
ScopePair = tuple[str | None, str | None]


def _mentioned_pairs(mentioned_companies: set[str], mentioned_periods: set[str]) -> set[ScopePair]:
    if mentioned_companies and mentioned_periods:
        return {(c, p) for c in mentioned_companies for p in mentioned_periods}
    if mentioned_companies:
        return {(c, None) for c in mentioned_companies}
    if mentioned_periods:
        return {(None, p) for p in mentioned_periods}
    return set()


def _pair_is_covered(pair: ScopePair, documents: list[Document]) -> bool:
    company, period = pair
    return any(
        (company is None or doc.metadata.get("company") == company)
        and (period is None or doc.metadata.get("period") == period)
        for doc in documents
    )


def _format_reason(missing_pairs: list[ScopePair], repair_attempted: bool) -> str:
    parts = []
    for company, period in missing_pairs:
        if company and period:
            parts.append(f"{company} {period}")
        elif company:
            parts.append(f"company {company}")
        else:
            parts.append(f"period {period}")
    reason = f"The retrieved evidence does not cover {', '.join(parts)} named in the question."
    if repair_attempted:
        reason += " A targeted retrieval for the missing coverage was already attempted."
    return reason


def find_missing_pairs(
    question: str, documents: list[Document], company_aliases: dict[str, list[str]]
) -> list[ScopePair]:
    """(company, period) pairs the question names but `documents` don't
    cover. Empty if the question names nothing specific, or everything
    named is covered.
    """
    mentioned_companies = extract_mentioned_companies(question, company_aliases)
    mentioned_periods = extract_mentioned_periods(question)
    pairs = _mentioned_pairs(mentioned_companies, mentioned_periods)
    missing = [p for p in pairs if not _pair_is_covered(p, documents)]
    # Sort key coerces None to "" -- pairs are structurally uniform within a
    # single call (see _mentioned_pairs), but this stays safe even if that
    # ever changes, rather than risking a str/NoneType comparison crash.
    return sorted(missing, key=lambda p: (p[0] or "", p[1] or ""))


@dataclass
class ScopeRepairResult:
    # Same shape as retrieval's own `RetrievalResult.reranked`, unchanged
    # (original) if no gap was found or repair wasn't needed, or the
    # merged-and-reranked set if a repair attempt ran. Carrying this shape
    # rather than a bare top score lets the caller feed it straight into
    # the existing score-threshold gate (`should_refuse_on_retrieval`)
    # exactly as before -- that gate's own code doesn't change at all.
    reranked: list[tuple[Document, float]]
    refusal_reason: str | None  # None if scope is covered (originally or after repair)


def attempt_scope_repair(
    question: str,
    reranked: list[tuple[Document, float]],
    company_aliases: dict[str, list[str]],
) -> ScopeRepairResult:
    """Check scope coverage; if there's a gap, make exactly one targeted
    retrieval attempt per missing pair before giving up.
    """
    documents = [doc for doc, _ in reranked]
    missing = find_missing_pairs(question, documents, company_aliases)
    if not missing:
        return ScopeRepairResult(reranked=reranked, refusal_reason=None)

    merged = list(documents)
    seen_chunk_ids = {doc.metadata.get("chunk_id") for doc in merged}
    for company, period in missing:
        targeted = retrieve(question, company=company, period=period)
        for doc, _ in targeted.reranked:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in seen_chunk_ids:
                merged.append(doc)
                seen_chunk_ids.add(chunk_id)

    repaired_reranked = rerank(question, merged, settings.rerank_top_k)
    repaired_documents = [doc for doc, _ in repaired_reranked]

    still_missing = find_missing_pairs(question, repaired_documents, company_aliases)
    if not still_missing:
        return ScopeRepairResult(reranked=repaired_reranked, refusal_reason=None)

    return ScopeRepairResult(
        reranked=repaired_reranked,
        refusal_reason=_format_reason(still_missing, repair_attempted=True),
    )
