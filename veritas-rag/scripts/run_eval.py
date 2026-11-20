"""Run the continuous-evaluation suite and print a report.

Exit code 1 when any hallucination or adversarial failure is detected, so
this doubles as a CI gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.harness import EvalCase, run_full_eval
from app.ingestion.versioning import make_doc_id
from app.pipeline import RagPipeline

CORPUS = {
    "data-retention.md": (
        "Customer records are retained for seven years after account closure. "
        "Backups are encrypted with AES-256 at rest. "
        "Deletion requests are honored within 30 days."
    ),
    "sla.txt": (
        "The service level agreement guarantees 99.95 percent monthly uptime. "
        "Credits of 10 percent apply when uptime falls below the guarantee. "
        "Scheduled maintenance windows are excluded from uptime calculations."
    ),
}

CASES = [
    EvalCase(
        query="How long are customer records retained after closure?",
        expected_doc_id=make_doc_id("data-retention.md", 1),
        expected_phrase="seven years",
    ),
    EvalCase(
        query="What uptime does the service level agreement guarantee?",
        expected_doc_id=make_doc_id("sla.txt", 1),
        expected_phrase="99.95",
    ),
]

UNANSWERABLE = [
    "What is the CEO's favorite color?",
    "How many satellites does the company operate?",
    "What is the recommended tire pressure for the delivery vans?",
]


def main() -> int:
    pipeline = RagPipeline()
    for source, text in CORPUS.items():
        pipeline.ingest(text.encode("utf-8"), source=source)

    report = run_full_eval(pipeline, CASES, UNANSWERABLE)
    print(json.dumps(report.to_dict(), indent=2))

    failed = report.hallucination_rate > 0 or report.adversarial_pass_rate < 1.0
    print(
        f"\nrecall@k={report.recall_at_k} accuracy={report.answer_accuracy} "
        f"hallucination_rate={report.hallucination_rate} "
        f"adversarial_pass_rate={report.adversarial_pass_rate}"
    )
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
