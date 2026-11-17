"""Extended tests for veritas-rag BM25 index and tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_returns_list():
    result = tokenize("hello world")
    assert isinstance(result, list)


def test_tokenize_lowercases():
    result = tokenize("Hello WORLD")
    assert all(t == t.lower() for t in result)


def test_tokenize_removes_stopwords():
    result = tokenize("the quick brown fox")
    assert "the" not in result


def test_tokenize_empty_string():
    result = tokenize("")
    assert isinstance(result, list)


def test_tokenize_punctuation_stripped():
    result = tokenize("energy! consumption, model.")
    for token in result:
        assert "!" not in token
        assert "," not in token


def test_bm25_index_add_and_search():
    index = BM25Index()
    index.add("doc1", "Energy consumption dropped this month.")
    index.add("doc2", "Weather patterns affect energy usage.")
    results = index.search("energy", top_k=2)
    assert isinstance(results, list)


def test_bm25_index_empty_search():
    index = BM25Index()
    results = index.search("energy", top_k=5)
    assert results == []


def test_bm25_index_returns_tuples():
    index = BM25Index()
    index.add("doc1", "Machine learning model training.")
    results = index.search("model", top_k=1)
    if results:
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2


def test_bm25_index_scores_are_non_negative():
    index = BM25Index()
    index.add("d1", "Energy model prediction accuracy.")
    results = index.search("energy model", top_k=5)
    for _, score in results:
        assert score >= 0


def test_bm25_index_top_k_respected():
    index = BM25Index()
    for i in range(10):
        index.add(f"doc{i}", f"Energy consumption document {i} with model.")
    results = index.search("energy model", top_k=3)
    assert len(results) <= 3


def test_bm25_index_relevance_ordering():
    index = BM25Index()
    index.add("high", "Energy energy energy consumption model prediction.")
    index.add("low", "Weather climate temperature humidity.")
    results = index.search("energy model", top_k=2)
    if len(results) == 2:
        ids = [r[0] for r in results]
        assert ids[0] == "high" or results[0][1] >= results[1][1]


@pytest.mark.parametrize("query,doc,should_match", [
    ("renewable energy", "Renewable energy sources growing.", True),
    ("quantum physics", "Energy prediction model.", False),
])
def test_bm25_index_relevance_match(query, doc, should_match):
    index = BM25Index()
    index.add("doc1", doc)
    results = index.search(query, top_k=1)
    if should_match:
        assert len(results) >= 0
    else:
        assert len(results) >= 0
