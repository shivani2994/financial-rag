"""
The embedding function shared by index-building and querying.

Wraps bge-base-en-v1.5 (via sentence-transformers, per the finalized stack)
in LangChain's HuggingFaceEmbeddings so it plugs directly into Chroma and,
later, retrieval. BGE models are trained to expect a short instruction
prefix on the *query* side only (not on stored passages) to get good
retrieval performance -- that's what `query_instruction` below is for.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        query_encode_kwargs={"prompt": BGE_QUERY_INSTRUCTION},
    )
