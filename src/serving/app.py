"""
The FastAPI service: wraps retrieval (Module 4) through generation
(Module 5) behind one endpoint.

Run with:

    uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Then POST to /ask, or open /docs for interactive Swagger docs.
"""

from dataclasses import asdict

from fastapi import FastAPI

from src.generation.pipeline import answer_question
from src.serving.schemas import AnswerResponse, CitationResponse, QuestionRequest

app = FastAPI(
    title="Financial RAG",
    description=(
        "Grounded, cited question answering over SEC filings and "
        "earnings-call transcripts for KO, PEP, and MDLZ. Refuses rather "
        "than guessing when retrieved context is weak."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    answer = answer_question(request.question, company=request.company, period=request.period)
    return AnswerResponse(
        question=answer.question,
        refused=answer.refused,
        reason=answer.reason,
        answer=answer.answer_text,
        citations=[CitationResponse(**asdict(c)) for c in answer.citations],
        confidence=answer.top_retrieval_score,
    )
