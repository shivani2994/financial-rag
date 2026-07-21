"""
The refusal gate: declines to answer when retrieval itself wasn't
confident, instead of handing weak context to the LLM and hoping it
hedges correctly.

Checked *before* calling the LLM at all -- weak retrieval never needs an
inference call, and no answer generated from weak context can be presented
as grounded regardless of what the model does with it.
"""

from langchain_core.documents import Document


def should_refuse_on_retrieval(
    reranked: list[tuple[Document, float]], threshold: float
) -> bool:
    """True if retrieval's top reranked score doesn't clear `threshold`,
    or if nothing was retrieved at all.
    """
    if not reranked:
        return True
    top_score = reranked[0][1]
    return top_score < threshold
