"""Building similarity index tests."""

from __future__ import annotations

from app.similarity import BuildingSimilarityIndex


def test_empty_index_returns_empty():
    idx = BuildingSimilarityIndex()
    assert idx.search([1.0, 2.0, 3.0]) == []


def test_add_and_search():
    idx = BuildingSimilarityIndex()
    idx.add("A", [1.0, 0.0, 0.0])
    idx.add("B", [0.0, 1.0, 0.0])
    results = idx.search([1.0, 0.0, 0.0], k=1)
    assert results[0][0] == "A"
    assert results[0][1] > 0.9


def test_similarity_ordering():
    idx = BuildingSimilarityIndex()
    idx.add("close", [1.0, 1.0, 0.0])
    idx.add("far", [0.0, 0.0, 1.0])
    results = idx.search([1.0, 1.0, 0.0])
    assert results[0][0] == "close"


def test_len():
    idx = BuildingSimilarityIndex()
    idx.add("x", [1.0])
    idx.add("y", [2.0])
    assert len(idx) == 2


def test_k_limit():
    idx = BuildingSimilarityIndex()
    for i in range(10):
        idx.add(f"b{i}", [float(i), float(i + 1)])
    results = idx.search([5.0, 6.0], k=3)
    assert len(results) == 3
