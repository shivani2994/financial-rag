"""
RAGAS evaluation harness.

Run with:

    uv run python -m src.evaluation.harness

Runs every question in eval/question_set.xlsx through the full pipeline
(retrieval through generation, Module 4 + Module 5), then:

- Scores every *answered* question with RAGAS: faithfulness, answer
  relevance, and context precision. All three are judged by the same local
  Ollama model and bge embeddings already used elsewhere in this project --
  no paid API calls, consistent with the open-source-only hard rule.
  Context precision uses RAGAS's reference-free variant
  (LLMContextPrecisionWithoutReference), since the question set gives a
  rubric ("what a good answer contains"), not a full ground-truth answer,
  and treating a rubric as a ground truth would misrepresent the metric.
- Reports refusal accuracy on the sheet's dedicated "5 - Refusal test"
  questions: the fraction correctly refused.
- Logs every query (Module 7's observability half) to
  `settings.query_log_path` via `log_query` -- latency and top retrieved
  sources, refused or not.
- Appends one run-summary line (the aggregate scores plus the run's
  configuration) to `settings.eval_run_log_path`, so a baseline isn't lost
  the moment the terminal scrolls past it.

Refused questions are excluded from RAGAS scoring: there's no generated
answer to judge faithfulness/relevance/precision against, by definition.

The RAGAS judge model (`settings.ragas_judge_model`) is deliberately a
*separate* setting from the generation model under test
(`settings.ollama_model`) and stays fixed across comparison runs -- if the
judge changed along with the generation model, a score difference could be
the judge behaving differently rather than generation improving. Swap
`OLLAMA_MODEL` (env var or .env) to compare generation models; the judge
doesn't move.
"""

import time

from langchain_ollama import OllamaLLM
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference
from ragas.run_config import RunConfig

from config.settings import settings
from src.evaluation.logging_setup import log_query, log_run_summary
from src.evaluation.question_set import EvalQuestion, load_question_set
from src.generation.pipeline import Answer, answer_question
from src.indexing.embeddings import get_embedding_function


def _judge_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(
        OllamaLLM(
            model=settings.ragas_judge_model, base_url=settings.ollama_base_url, temperature=0.0
        )
    )


def run_evaluation() -> tuple[
    list[EvalQuestion], list[tuple[str, bool]], list[Answer], object | None, float | None
]:
    questions = load_question_set(settings.eval_question_set_path)

    refusal_results: list[tuple[str, bool]] = []  # (question_id, was_refused)
    answers: list[Answer] = []
    samples = []

    for q in questions:
        start = time.perf_counter()
        answer = answer_question(q.question)
        latency = time.perf_counter() - start

        log_query(settings.query_log_path, answer, latency, settings.top_sources_logged)
        answers.append(answer)

        if q.is_refusal_question:
            refusal_results.append((q.id, answer.refused))

        if not answer.refused:
            samples.append(
                SingleTurnSample(
                    user_input=q.question,
                    response=answer.answer_text,
                    retrieved_contexts=[doc.page_content for doc in answer.retrieved_documents],
                )
            )

    ragas_result = None
    if samples:
        # RAGAS's defaults (180s timeout, 16 concurrent workers) assume a
        # hosted API; a small local model answering sequentially through
        # one Ollama instance needs a much longer per-job timeout and far
        # less concurrency, or jobs queue past the timeout and fail as
        # TimeoutError with nan scores (confirmed during Module 7 testing).
        ragas_result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference()],
            llm=_judge_llm(),
            embeddings=LangchainEmbeddingsWrapper(get_embedding_function()),
            run_config=RunConfig(timeout=600, max_workers=2),
        )

    refusal_accuracy = None
    if refusal_results:
        refusal_accuracy = sum(1 for _, r in refusal_results if r) / len(refusal_results)

    answered_count = sum(1 for a in answers if not a.refused)
    refused_count = sum(1 for a in answers if a.refused)
    log_run_summary(
        settings.eval_run_log_path,
        ragas_result,
        refusal_accuracy,
        answered_count,
        refused_count,
        len(questions),
    )

    return questions, refusal_results, answers, ragas_result, refusal_accuracy


def _print_report(
    questions: list[EvalQuestion],
    refusal_results: list[tuple[str, bool]],
    answers: list[Answer],
    ragas_result: object | None,
    refusal_accuracy: float | None,
) -> None:
    print(f"Ran {len(questions)} questions from {settings.eval_question_set_path}")
    print(f"Generation model: {settings.ollama_model}  |  RAGAS judge: {settings.ragas_judge_model}")
    print(f"Answered: {sum(1 for a in answers if not a.refused)}  "
          f"Refused: {sum(1 for a in answers if a.refused)}")

    print("\n--- RAGAS scores (answered questions only) ---")
    if ragas_result is not None:
        print(ragas_result)
    else:
        print("No answered questions to score.")

    print("\n--- Refusal accuracy (scenario '5 - Refusal test') ---")
    if refusal_results:
        correct = sum(1 for _, refused in refusal_results if refused)
        print(f"{correct}/{len(refusal_results)} correctly refused ({refusal_accuracy:.0%})")
        for qid, refused in refusal_results:
            print(f"  {qid}: {'refused (correct)' if refused else 'ANSWERED (incorrect)'}")
    else:
        print("No refusal-test questions found in the sheet.")

    print(f"\nPer-query logs: {settings.query_log_path}")
    print(f"Run summary logged to: {settings.eval_run_log_path}")


if __name__ == "__main__":
    questions, refusal_results, answers, ragas_result, refusal_accuracy = run_evaluation()
    _print_report(questions, refusal_results, answers, ragas_result, refusal_accuracy)
