"""Evaluation harness: recall, hallucination rate, adversarial resistance.

Run continuously (CI + scheduled) so retrieval or grounding regressions are
caught the moment they land, not when a user notices a made-up answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..pipeline import RagPipeline
from .adversarial import ADVERSARIAL_QUERIES


@dataclass
class EvalCase:
    """A gold question with its expected evidence.

    Attributes:
        query: The question.
        expected_doc_id: Document that must appear in the citations.
        expected_phrase: Phrase the answer must contain (empty = any answer).
    """

    query: str
    expected_doc_id: str
    expected_phrase: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation results."""

    recall_at_k: float = 0.0
    answer_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    adversarial_pass_rate: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "answer_accuracy": self.answer_accuracy,
            "hallucination_rate": self.hallucination_rate,
            "adversarial_pass_rate": self.adversarial_pass_rate,
            "details": self.details,
        }


def run_recall_eval(
    pipeline: RagPipeline, cases: list[EvalCase]
) -> tuple[float, float, list[dict[str, Any]]]:
    """Measure retrieval recall and answer accuracy on gold cases.

    Recall@k: the expected document appears among the cited evidence.
    Accuracy: the answer contains the expected phrase.
    """
    details: list[dict[str, Any]] = []
    recall_hits = 0
    accuracy_hits = 0

    for case in cases:
        answer = pipeline.query(case.query)
        cited_docs = {c.doc_id for c in answer.citations}
        recalled = case.expected_doc_id in cited_docs
        accurate = bool(
            answer.answered
            and (not case.expected_phrase or case.expected_phrase.lower() in answer.text.lower())
        )
        recall_hits += recalled
        accuracy_hits += accurate
        details.append(
            {
                "kind": "recall",
                "query": case.query,
                "recalled": recalled,
                "accurate": accurate,
                "answered": answer.answered,
                "confidence": answer.confidence,
            }
        )

    n = max(1, len(cases))
    return recall_hits / n, accuracy_hits / n, details


def run_hallucination_eval(
    pipeline: RagPipeline, unanswerable_queries: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    """Measure the hallucination rate on questions the corpus cannot answer.

    Any confident answer to an unanswerable question is a hallucination;
    the correct behavior is the insufficient-evidence fallback.
    """
    details: list[dict[str, Any]] = []
    hallucinations = 0
    for query in unanswerable_queries:
        answer = pipeline.query(query)
        hallucinated = answer.answered
        hallucinations += hallucinated
        details.append(
            {
                "kind": "hallucination",
                "query": query,
                "hallucinated": hallucinated,
                "confidence": answer.confidence,
            }
        )
    return hallucinations / max(1, len(unanswerable_queries)), details


def run_adversarial_eval(pipeline: RagPipeline) -> tuple[float, list[dict[str, Any]]]:
    """Run the adversarial suite; a case passes when no ungrounded answer
    is produced (either fallback fires, or every claim carries a citation)."""
    details: list[dict[str, Any]] = []
    passes = 0
    for case in ADVERSARIAL_QUERIES:
        answer = pipeline.query(case["query"])
        grounded = (not answer.answered) or bool(answer.citations)
        passes += grounded
        details.append(
            {
                "kind": "adversarial",
                "name": case["name"],
                "query": case["query"],
                "passed": grounded,
                "answered": answer.answered,
            }
        )
    return passes / max(1, len(ADVERSARIAL_QUERIES)), details


def run_full_eval(
    pipeline: RagPipeline,
    cases: list[EvalCase],
    unanswerable_queries: list[str],
) -> EvalReport:
    """Run every evaluation mode and aggregate a report."""
    recall, accuracy, recall_details = run_recall_eval(pipeline, cases)
    halluc_rate, halluc_details = run_hallucination_eval(pipeline, unanswerable_queries)
    adv_rate, adv_details = run_adversarial_eval(pipeline)

    return EvalReport(
        recall_at_k=round(recall, 4),
        answer_accuracy=round(accuracy, 4),
        hallucination_rate=round(halluc_rate, 4),
        adversarial_pass_rate=round(adv_rate, 4),
        details=recall_details + halluc_details + adv_details,
    )
