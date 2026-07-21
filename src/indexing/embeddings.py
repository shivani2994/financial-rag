"""
The embedding function shared by index-building and querying.

Wraps bge-base-en-v1.5 (via sentence-transformers, per the finalized stack)
in LangChain's HuggingFaceEmbeddings so it plugs directly into Chroma and,
later, retrieval. BGE models are trained to expect a short instruction
prefix on the *query* side only (not on stored passages) to get good
retrieval performance -- that's what `query_instruction` below is for.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Cached: a long-lived server (Module 6) calls this on every request,
    and reloading the model's weights from disk each time would make every
    request pay multiple seconds of load time for no reason -- the model
    itself is read-only and safe to share.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        query_encode_kwargs={"prompt": BGE_QUERY_INSTRUCTION},
    )
