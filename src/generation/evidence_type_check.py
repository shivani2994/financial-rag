"""
Deterministic evidence-type check.

Some questions ask for evidence this corpus can never contain -- live or
current market data -- no matter how good retrieval or the LLM are. Caught
here with a fixed keyword rule set, before any LLM call and independent of
retrieval, so the answer is always "refused" for these regardless of
reranker score. This is what a reranker false positive can't override: the
evaluation harness surfaced exactly this gap on "What is Coca-Cola's stock
price today?", which scored a deceptively high 0.999 (the reranker matched
it to a 10-K's "Market for Registrant's Common Equity" section, which
discusses the stock but not a live price) and got answered instead of
refused.

Pure string matching: no LLM call, no randomness, identical output for the
same question on every run.
"""


def check_evidence_type(question: str, live_market_data_keywords: list[str]) -> str | None:
    """Return a refusal reason if the question asks for live/current market
    data; None if it doesn't match this rule set.
    """
    question_lower = question.lower()
    for keyword in live_market_data_keywords:
        if keyword.lower() in question_lower:
            return (
                f"This question asks for live/current market data (matched "
                f"phrase {keyword!r}), which this corpus of static SEC "
                f"filings and transcripts does not contain."
            )
    return None
