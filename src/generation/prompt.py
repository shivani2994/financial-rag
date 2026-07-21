"""Builds the grounding prompt: numbered, source-labeled context passages,
plus instructions that force the model to answer only from them and cite
every claim by passage number.
"""

from langchain_core.documents import Document

NO_ANSWER_MARKER = "INSUFFICIENT_CONTEXT"

_TEMPLATE = """You are a financial research assistant. Answer the question using ONLY \
the numbered context passages below. Do not use any outside knowledge, even \
if you know the answer -- if it isn't in the passages, it doesn't exist for \
this task.

Rules:
1. Every factual claim in your answer must end with a citation marker like \
[1] or [2] naming the passage(s) it came from.
2. If the passages do not contain enough information to answer the \
question, respond with exactly: {no_answer_marker}
3. Do not fabricate a citation number that isn't listed below.

Context passages:
{context}

Question: {question}

Answer (with citation markers):"""


def _format_context(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        source = f"{meta.get('company')}, {meta.get('doc_type')}, {meta.get('period')}"
        section_or_speaker = meta.get("section_or_speaker")
        if section_or_speaker:
            source += f", {section_or_speaker}"
        blocks.append(f"[{i}] ({source})\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, documents: list[Document]) -> str:
    return _TEMPLATE.format(
        no_answer_marker=NO_ANSWER_MARKER,
        context=_format_context(documents),
        question=question,
    )
