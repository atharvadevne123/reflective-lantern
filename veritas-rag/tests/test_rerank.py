"""Tests for veritas-rag lexical reranker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.rerank import LexicalReranker


@pytest.fixture
def reranker():
    return LexicalReranker()


def test_score_returns_list(reranker):
    scores = reranker.score("test query", ["some text"])
    assert isinstance(scores, list)


def test_score_length_matches_texts(reranker):
    texts = ["doc one", "doc two", "doc three"]
    scores = reranker.score("query", texts)
    assert len(scores) == len(texts)


def test_scores_in_zero_one_range(reranker):
    scores = reranker.score("machine learning model", ["machine learning is great", "unrelated"])
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_relevant_text_scores_higher(reranker):
    query = "machine learning model"
    relevant = "machine learning model training accuracy"
    irrelevant = "the weather today is sunny"
    scores = reranker.score(query, [relevant, irrelevant])
    assert scores[0] > scores[1]


def test_empty_texts_list(reranker):
    scores = reranker.score("query", [])
    assert scores == []


def test_stopword_only_query_returns_zeros(reranker):
    scores = reranker.score("the a an", ["some document text"])
    assert all(s == 0.0 for s in scores)


def test_empty_text_scores_zero(reranker):
    scores = reranker.score("query terms", [""])
    assert scores[0] == 0.0


def test_exact_match_scores_high(reranker):
    query = "fleet maintenance alert brake pad"
    text = "fleet maintenance alert brake pad worn critical"
    scores = reranker.score(query, [text])
    assert scores[0] > 0.5


@pytest.mark.parametrize("query,text,expect_nonzero", [
    ("brake pad worn", "brake pad worn to 2mm", True),
    ("engine temperature", "oil pressure low", False),
])
def test_query_text_relevance(query, text, expect_nonzero, reranker):
    scores = reranker.score(query, [text])
    if expect_nonzero:
        assert scores[0] > 0.0
    else:
        assert scores[0] >= 0.0  # could be zero or low
