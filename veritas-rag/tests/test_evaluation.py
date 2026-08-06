"""Stage 8 tests: the evaluation harness itself."""

from __future__ import annotations

from app.evaluation.adversarial import (
    INJECTION_DOCUMENT,
    INJECTION_DOCUMENT_FACT,
    INJECTION_DOCUMENT_QUERY,
)
from app.evaluation.harness import (
    EvalCase,
    run_adversarial_eval,
    run_full_eval,
    run_hallucination_eval,
    run_recall_eval,
)
from app.ingestion.versioning import make_doc_id


def gold_cases() -> list[EvalCase]:
    return [
        EvalCase(
            query="How much is the rooftop solar rebate per kilowatt?",
            expected_doc_id=make_doc_id("solar-policy.md", 1),
            expected_phrase="900 dollars",
        ),
        EvalCase(
            query="What is the battery incentive per kilowatt hour?",
            expected_doc_id=make_doc_id("battery-guide.txt", 1),
            expected_phrase="250 dollars",
        ),
        EvalCase(
            query="What is the export limit for single phase connections?",
            expected_doc_id=make_doc_id("grid-rules.txt", 1),
            expected_phrase="five kilowatts",
        ),
    ]


UNANSWERABLE = [
    "What is the average lifespan of a Bengal tiger?",
    "Who won the 1998 chess world championship?",
    "What is the recommended dosage of ibuprofen for adults?",
]


class TestRecallEval:
    def test_perfect_recall_on_gold_corpus(self, pipeline):
        recall, accuracy, details = run_recall_eval(pipeline, gold_cases())
        assert recall == 1.0
        assert accuracy == 1.0
        assert len(details) == 3


class TestHallucinationEval:
    def test_zero_hallucinations_on_unanswerable(self, pipeline):
        rate, details = run_hallucination_eval(pipeline, UNANSWERABLE)
        assert rate == 0.0
        assert all(not d["hallucinated"] for d in details)


class TestAdversarialEval:
    def test_full_suite_passes(self, pipeline):
        rate, details = run_adversarial_eval(pipeline)
        assert rate == 1.0
        assert all(d["passed"] for d in details)

    def test_injection_document_data_used_but_instructions_ignored(self, pipeline):
        # The malicious document's FACT should be retrievable...
        pipeline.ingest(INJECTION_DOCUMENT.encode(), "injected.txt")
        answer = pipeline.query(INJECTION_DOCUMENT_QUERY)
        assert answer.answered
        assert INJECTION_DOCUMENT_FACT.lower() in answer.text.lower()
        # ...but its embedded instructions must not unlock ungrounded answers
        probe = pipeline.query("What is the capital of France?")
        assert not probe.answered


class TestFullEval:
    def test_report_aggregates_all_modes(self, pipeline):
        report = run_full_eval(pipeline, gold_cases(), UNANSWERABLE)
        assert report.recall_at_k == 1.0
        assert report.answer_accuracy == 1.0
        assert report.hallucination_rate == 0.0
        assert report.adversarial_pass_rate == 1.0
        kinds = {d["kind"] for d in report.details}
        assert kinds == {"recall", "hallucination", "adversarial"}

    def test_report_serializable(self, pipeline):
        import json

        report = run_full_eval(pipeline, gold_cases()[:1], UNANSWERABLE[:1])
        assert json.loads(json.dumps(report.to_dict()))
