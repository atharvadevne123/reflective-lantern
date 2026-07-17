"""End-to-end demo: ingest a corpus, ask questions, inspect a trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import RagPipeline

CORPUS = {
    "onboarding-policy.md": (
        "# Onboarding Policy\n\n"
        "New employees receive laptop hardware within five business days. "
        "Security training must be completed within the first two weeks. "
        "Building access badges are issued by the facilities team on day one."
    ),
    "expense-rules.txt": (
        "Meal expenses are reimbursed up to 60 dollars per day of travel. "
        "Flights above 800 dollars require prior manager approval. "
        "Expense reports must be submitted within 30 days of the trip."
    ),
}

QUESTIONS = [
    "How quickly do new employees receive laptops?",
    "What is the daily meal reimbursement limit?",
    "What is the company policy on remote work from Antarctica?",  # unanswerable
]


def main() -> None:
    pipeline = RagPipeline()

    print("=== Ingest ===")
    for source, text in CORPUS.items():
        report = pipeline.ingest(text.encode("utf-8"), source=source)
        print(f"  {source}: {report['status']} ({report.get('chunks', 0)} chunks)")

    print("\n=== Query ===")
    last_request_id = ""
    for question in QUESTIONS:
        answer = pipeline.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer.text}")
        print(f"   answered={answer.answered} confidence={answer.confidence}")
        for cite in answer.citations:
            print(f"   [cite] {cite.source} p.{cite.page}: {cite.quote[:60]}...")
        last_request_id = answer.request_id

    print("\n=== Trace of the last request ===")
    trace = pipeline.traces.get(last_request_id)
    if trace:
        print(json.dumps(trace.to_dict(), indent=2)[:2000])

    print("\n=== Metrics ===")
    print(json.dumps(pipeline.metrics(), indent=2))


if __name__ == "__main__":
    main()
