"""
Fast, deterministic evaluation gate -- the CI-friendly counterpart to
`src/evaluation/harness.py`'s full RAGAS run.

Runs all 15 questions in eval/question_set.xlsx through *real* retrieval
(dense + BM25 + bge-reranker-base, exactly as production does) and the
three pre-generation refusal gates that need no LLM call: the evidence-type
check, the scope-coverage check (with its bounded repair attempt), and the
score-threshold check. It stops there -- it never calls the LLM. That means
it cannot see the two generation-dependent refusal points (the model's own
self-refusal marker, and citation enforcement), so a question that clears
all three gates here is reported as "accepted" (would proceed to
generation), not as "answered with a verified-grounded response". Full
answer quality is still only measured by the slow RAGAS harness.

This mode exists because the full harness needs a local LLM as a RAGAS
judge and takes 1.5-2.5 hours (see harness.py's own docstring) -- far too
slow to run on every push. This mode needs no LLM at all: retrieval and all
three gates it exercises are deterministic functions of the persisted
indexes and the question text, so it finishes in minutes and returns the
same verdict every time for the same code and data (verified: two
back-to-back local runs produced a byte-identical JSON summary).

Every check called here (`check_evidence_type`, `attempt_scope_repair`,
`should_refuse_on_retrieval`) is imported directly from
src/generation/{evidence_type_check,scope_check,refusal}.py -- the exact
same functions src/generation/pipeline.py calls, in the exact same order.
Nothing is reimplemented, so this gate can't silently drift from what the
real pipeline does up to the point it stops.

Run with:

    uv run python -m src.evaluation.fast_gate

Writes a JSON summary to `settings.fast_gate_result_path` and exits 1 (a
CI build failure) if refusal accuracy or overall decision accuracy falls
below the thresholds in `settings.gate_thresholds_path`.
"""

import json
import sys
from dataclasses import asdict, dataclass

from config.settings import settings
from src.evaluation.question_set import EvalQuestion, load_question_set
from src.generation.evidence_type_check import check_evidence_type
from src.generation.refusal import should_refuse_on_retrieval
from src.generation.scope_check import attempt_scope_repair
from src.retrieval.pipeline import retrieve


@dataclass
class GateDecision:
    question_id: str
    question: str
    is_refusal_question: bool
    accepted: bool  # cleared all three pre-generation gates -> would reach the LLM
    refusal_reason: str | None
    top_retrieval_score: float | None


def evaluate_pre_generation_gates(question: str) -> tuple[bool, str | None, float | None]:
    """Run retrieval plus the three deterministic pre-generation gates for
    one question -- mirrors `src/generation/pipeline.py::answer_question`
    exactly, up to (not including) the LLM call.

    Returns (accepted, refusal_reason, top_retrieval_score).
    """
    result = retrieve(question)
    top_score = float(result.reranked[0][1]) if result.reranked else None

    evidence_type_reason = check_evidence_type(question, settings.live_market_data_keywords)
    if evidence_type_reason is not None:
        return False, evidence_type_reason, top_score

    scope_repair = attempt_scope_repair(question, result.reranked, settings.company_aliases)
    reranked = scope_repair.reranked
    top_score = float(reranked[0][1]) if reranked else None
    if scope_repair.refusal_reason is not None:
        return False, scope_repair.refusal_reason, top_score

    if should_refuse_on_retrieval(reranked, settings.refusal_confidence_threshold):
        reason = (
            f"No retrieved passage was confident enough to ground an answer "
            f"(top rerank score {top_score:.4f} < threshold "
            f"{settings.refusal_confidence_threshold})"
            if top_score is not None
            else "No passages were retrieved for this question."
        )
        return False, reason, top_score

    return True, None, top_score


def run_fast_gate(questions: list[EvalQuestion]) -> dict:
    decisions = []
    for q in questions:
        accepted, reason, score = evaluate_pre_generation_gates(q.question)
        decisions.append(
            GateDecision(
                question_id=q.id,
                question=q.question,
                is_refusal_question=q.is_refusal_question,
                accepted=accepted,
                refusal_reason=reason,
                top_retrieval_score=score,
            )
        )

    answerable = [d for d in decisions if not d.is_refusal_question]
    unanswerable = [d for d in decisions if d.is_refusal_question]
    answerable_accepted = sum(1 for d in answerable if d.accepted)
    unanswerable_refused = sum(1 for d in unanswerable if not d.accepted)
    refusal_accuracy = unanswerable_refused / len(unanswerable) if unanswerable else None
    decision_accuracy = (
        (answerable_accepted + unanswerable_refused) / len(decisions) if decisions else None
    )

    return {
        "num_questions": len(decisions),
        "answerable_accepted": answerable_accepted,
        "answerable_total": len(answerable),
        "unanswerable_refused": unanswerable_refused,
        "unanswerable_total": len(unanswerable),
        "refusal_accuracy": refusal_accuracy,
        "decision_accuracy": decision_accuracy,
        "decisions": [asdict(d) for d in decisions],
    }


def _check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    failures = []
    if (
        summary["refusal_accuracy"] is not None
        and summary["refusal_accuracy"] < thresholds["min_refusal_accuracy"]
    ):
        failures.append(
            f"refusal_accuracy {summary['refusal_accuracy']:.3f} < "
            f"threshold {thresholds['min_refusal_accuracy']}"
        )
    if (
        summary["decision_accuracy"] is not None
        and summary["decision_accuracy"] < thresholds["min_decision_accuracy"]
    ):
        failures.append(
            f"decision_accuracy {summary['decision_accuracy']:.3f} < "
            f"threshold {thresholds['min_decision_accuracy']}"
        )
    return failures


def main() -> int:
    questions = load_question_set(settings.eval_question_set_path)
    summary = run_fast_gate(questions)

    settings.fast_gate_result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.fast_gate_result_path, "w") as f:
        json.dump(summary, f, indent=2)

    headline = {k: v for k, v in summary.items() if k != "decisions"}
    print(json.dumps(headline, indent=2))

    with open(settings.gate_thresholds_path) as f:
        thresholds = json.load(f)

    failures = _check_thresholds(summary, thresholds)
    if failures:
        print("\nFAST GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nFast gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
