"""
Deterministic scope-coverage check.

Parses the question for named companies (tickers and known aliases) and
named periods (e.g. "Q1 2026", "2026Q1", "FY2025"), then compares that
scope against the company/period metadata actually carried on the reranked
chunks handed up from retrieval. If the question names an entity or period
the retrieved set doesn't cover, that's a real gap regardless of how
confident the reranker's scores looked -- the corpus may hold the answer,
but the evidence in hand doesn't, so refusing beats guessing.

Regex and set comparisons only: no LLM call, no randomness, identical
output for the same question and retrieved documents on every run.
"""

import re

from langchain_core.documents import Document

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


def check_scope_coverage(
    question: str,
    retrieved_documents: list[Document],
    company_aliases: dict[str, list[str]],
) -> str | None:
    """Return a refusal reason naming exactly which company/period the
    question asks about but the retrieved documents don't cover; None if
    the question names no specific company/period, or everything it names
    is covered.
    """
    mentioned_companies = extract_mentioned_companies(question, company_aliases)
    mentioned_periods = extract_mentioned_periods(question)
    if not mentioned_companies and not mentioned_periods:
        return None

    covered_companies = {doc.metadata.get("company") for doc in retrieved_documents}
    covered_periods = {doc.metadata.get("period") for doc in retrieved_documents}

    missing_companies = sorted(mentioned_companies - covered_companies)
    missing_periods = sorted(mentioned_periods - covered_periods)
    if not missing_companies and not missing_periods:
        return None

    gaps = []
    if missing_companies:
        gaps.append(f"company {', '.join(missing_companies)}")
    if missing_periods:
        gaps.append(f"period {', '.join(missing_periods)}")
    return f"The retrieved evidence does not cover the {' and '.join(gaps)} named in the question."
