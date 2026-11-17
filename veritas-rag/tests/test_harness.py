"""Tests for veritas-rag evaluation harness dataclasses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.harness import EvalCase, EvalReport


def test_eval_case_stores_query():
    case = EvalCase(query="What is energy?", expected_doc_id="doc1")
    assert case.query == "What is energy?"


def test_eval_case_stores_expected_doc_id():
    case = EvalCase(query="test", expected_doc_id="doc42")
    assert case.expected_doc_id == "doc42"


def test_eval_case_default_expected_phrase():
    case = EvalCase(query="test", expected_doc_id="doc1")
    assert case.expected_phrase == ""


def test_eval_case_custom_expected_phrase():
    case = EvalCase(query="test", expected_doc_id="doc1", expected_phrase="renewable energy")
    assert case.expected_phrase == "renewable energy"


def test_eval_report_default_values():
    report = EvalReport()
    assert report.recall_at_k == 0.0
    assert report.answer_accuracy == 0.0
    assert report.hallucination_rate == 0.0
    assert report.adversarial_pass_rate == 0.0
    assert report.details == []


def test_eval_report_to_dict_returns_dict():
    report = EvalReport(recall_at_k=0.8, answer_accuracy=0.75)
    result = report.to_dict()
    assert isinstance(result, dict)


def test_eval_report_to_dict_has_all_keys():
    report = EvalReport()
    result = report.to_dict()
    assert "recall_at_k" in result
    assert "answer_accuracy" in result
    assert "hallucination_rate" in result
    assert "adversarial_pass_rate" in result
    assert "details" in result


def test_eval_report_to_dict_preserves_values():
    report = EvalReport(recall_at_k=0.9, hallucination_rate=0.05)
    result = report.to_dict()
    assert result["recall_at_k"] == 0.9
    assert result["hallucination_rate"] == 0.05


def test_eval_report_details_is_list():
    report = EvalReport(details=[{"query": "q1", "hit": True}])
    assert isinstance(report.details, list)
    assert len(report.details) == 1


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_eval_report_rate_range(rate):
    report = EvalReport(recall_at_k=rate, adversarial_pass_rate=rate)
    d = report.to_dict()
    assert d["recall_at_k"] == rate
    assert d["adversarial_pass_rate"] == rate
