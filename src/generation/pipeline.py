"""
Generation pipeline: retrieval -> refusal gate -> grounded, cited answer.

Run with:

    uv run python -m src.generation.pipeline "your question" [--company KO] [--period 2026Q1]

Three refusal points, in order -- any one of them can end the pipeline
before returning an unfounded answer:

1. Before calling the LLM at all: if Module 4's top reranked score doesn't
   clear `refusal_confidence_threshold`, retrieval itself wasn't confident
   enough to ground an answer.
2. After calling the LLM: if the model itself says the passages don't
   answer the question (it was told to respond with a fixed marker rather
   than guess).
3. After parsing the answer: if the model's response doesn't cite any real
   passage, it can't be verified as grounded, so it's treated as a failure
   rather than presented as an answer.
"""

import argparse
from dataclasses import dataclass, field

from langchain_core.documents import Document

from config.settings import settings
from src.generation.citations import Citation, resolve_citations
from src.generation.llm import get_llm_client
from src.generation.prompt import NO_ANSWER_MARKER, build_prompt
from src.generation.refusal import should_refuse_on_retrieval
from src.retrieval.pipeline import retrieve


@dataclass
class Answer:
    question: str
    refused: bool
    reason: str | None
    answer_text: str | None
    citations: list[Citation] = field(default_factory=list)
    top_retrieval_score: float | None = None
    # The reranked passages retrieval handed to the LLM (populated even when
    # refused, so observability/evaluation can see what almost-but-didn't
    # clear the bar). Module 7 needs this for RAGAS's context-based metrics
    # and for logging each query's top retrieved sources.
    retrieved_documents: list[Document] = field(default_factory=list)


def answer_question(
    question: str,
    company: str | None = None,
    period: str | None = None,
) -> Answer:
    result = retrieve(question, company=company, period=period)
    top_score = float(result.reranked[0][1]) if result.reranked else None
    retrieved_documents = [doc for doc, _ in result.reranked]

    if should_refuse_on_retrieval(result.reranked, settings.refusal_confidence_threshold):
        reason = (
            f"No retrieved passage was confident enough to ground an answer "
            f"(top rerank score {top_score:.4f} < threshold "
            f"{settings.refusal_confidence_threshold})"
            if top_score is not None
            else "No passages were retrieved for this question."
        )
        return Answer(
            question=question, refused=True, reason=reason, answer_text=None,
            top_retrieval_score=top_score, retrieved_documents=retrieved_documents,
        )

    documents = retrieved_documents
    prompt = build_prompt(question, documents)
    raw_answer = get_llm_client().generate(prompt).strip()

    if NO_ANSWER_MARKER in raw_answer:
        return Answer(
            question=question,
            refused=True,
            reason="The model determined the retrieved passages don't answer the question.",
            answer_text=None,
            top_retrieval_score=top_score,
            retrieved_documents=retrieved_documents,
        )

    citations = resolve_citations(raw_answer, documents)
    if not citations:
        return Answer(
            question=question,
            refused=True,
            reason=(
                "The model's answer didn't cite any retrieved passage, so it "
                "can't be verified as grounded."
            ),
            answer_text=None,
            top_retrieval_score=top_score,
            retrieved_documents=retrieved_documents,
        )

    return Answer(
        question=question,
        refused=False,
        reason=None,
        answer_text=raw_answer,
        citations=citations,
        top_retrieval_score=top_score,
        retrieved_documents=retrieved_documents,
    )


def _print_answer(answer: Answer) -> None:
    print(f"Question: {answer.question!r}")
    print(f"Top retrieval score: {answer.top_retrieval_score}")
    if answer.refused:
        print(f"\nREFUSED: {answer.reason}")
        return

    print(f"\nAnswer:\n{answer.answer_text}")
    print(f"\n--- {len(answer.citations)} citation(s) ---")
    for c in answer.citations:
        source = f"{c.company}, {c.doc_type}, {c.period}"
        if c.section_or_speaker:
            source += f", {c.section_or_speaker}"
        print(f"[{c.marker}] {source}  (chunk_id={c.chunk_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Module 5 generation pipeline.")
    parser.add_argument("question")
    parser.add_argument("--company", default=None)
    parser.add_argument("--period", default=None)
    args = parser.parse_args()

    answer = answer_question(args.question, company=args.company, period=args.period)
    _print_answer(answer)
