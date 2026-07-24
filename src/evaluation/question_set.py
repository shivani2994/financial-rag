"""Reads the evaluation answer key from `eval/question_set.xlsx`."""

from dataclasses import dataclass
from pathlib import Path

import openpyxl

SHEET_NAME = "Evaluation Set"
HEADER_ROW = 4  # row 4 holds the column names; data starts row 5


@dataclass
class EvalQuestion:
    id: str
    scenario: str
    question: str
    expected_source: str | None
    good_answer_description: str | None
    is_refusal_question: bool


def load_question_set(path: Path) -> list[EvalQuestion]:
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[SHEET_NAME]

    questions = []
    for row in sheet.iter_rows(min_row=HEADER_ROW + 1, values_only=True):
        id_, scenario, question, expected_source, good_answer_description = row[:5]
        if id_ is None or str(id_).strip().upper() == "EXAMPLE":
            continue
        scenario = str(scenario) if scenario else ""
        questions.append(
            EvalQuestion(
                id=str(id_),
                scenario=scenario,
                question=str(question),
                expected_source=expected_source,
                good_answer_description=good_answer_description,
                is_refusal_question="refusal" in scenario.lower(),
            )
        )
    return questions
