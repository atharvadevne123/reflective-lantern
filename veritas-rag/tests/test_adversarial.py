"""Tests for veritas-rag adversarial query suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.adversarial import (
    ADVERSARIAL_QUERIES,
    INJECTION_DOCUMENT,
    INJECTION_DOCUMENT_FACT,
    INJECTION_DOCUMENT_QUERY,
)


def test_adversarial_queries_is_list():
    assert isinstance(ADVERSARIAL_QUERIES, list)


def test_adversarial_queries_not_empty():
    assert len(ADVERSARIAL_QUERIES) > 0


def test_each_query_has_required_keys():
    for item in ADVERSARIAL_QUERIES:
        assert "name" in item
        assert "query" in item
        assert "expectation" in item


def test_each_query_string_non_empty():
    for item in ADVERSARIAL_QUERIES:
        assert isinstance(item["query"], str)
        assert item["query"].strip()


def test_each_name_is_unique():
    names = [item["name"] for item in ADVERSARIAL_QUERIES]
    assert len(names) == len(set(names))


def test_injection_document_is_string():
    assert isinstance(INJECTION_DOCUMENT, str)
    assert len(INJECTION_DOCUMENT) > 0


def test_injection_document_query_is_string():
    assert isinstance(INJECTION_DOCUMENT_QUERY, str)


def test_injection_document_fact_is_string():
    assert isinstance(INJECTION_DOCUMENT_FACT, str)


def test_injection_document_contains_fact():
    assert INJECTION_DOCUMENT_FACT in INJECTION_DOCUMENT


@pytest.mark.parametrize("item", ADVERSARIAL_QUERIES)
def test_expectation_is_no_ungrounded_answer(item):
    assert item["expectation"] == "no_ungrounded_answer"
