"""Tests for veritas-rag approximate nearest-neighbour vector index."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.ann import VectorIndex


@pytest.fixture
def index():
    return VectorIndex(dim=16)


def _unit(dim=16):
    v = np.random.randn(1, dim).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_empty_index_len_is_zero(index):
    assert len(index) == 0


def test_add_increases_len(index):
    index.add(["c1"], _unit())
    assert len(index) == 1


def test_search_empty_returns_empty(index):
    q = _unit()[0]
    result = index.search(q)
    assert result == []


def test_search_returns_list(index):
    index.add(["c1"], _unit())
    q = _unit()[0]
    result = index.search(q)
    assert isinstance(result, list)


def test_search_top_k_respected(index):
    for i in range(10):
        index.add([f"c{i}"], _unit())
    q = _unit()[0]
    result = index.search(q, top_k=3)
    assert len(result) <= 3


def test_search_returns_tuples_of_id_and_score(index):
    index.add(["c1"], _unit())
    q = _unit()[0]
    result = index.search(q, top_k=1)
    for item in result:
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_remove_excludes_from_search(index):
    v = _unit()
    index.add(["c1"], v)
    index.add(["c2"], _unit())
    index.remove(["c1"])
    q = _unit()[0]
    result = index.search(q, top_k=10)
    ids = [r[0] for r in result]
    assert "c1" not in ids


def test_remove_decreases_len(index):
    index.add(["c1", "c2"], np.vstack([_unit(), _unit()]))
    index.remove(["c1"])
    assert len(index) == 1


def test_add_mismatched_raises(index):
    with pytest.raises(ValueError):
        index.add(["c1", "c2"], _unit())  # 1 vector, 2 ids


def test_exact_query_scores_highest(index):
    v = _unit()
    index.add(["target"], v)
    for i in range(5):
        index.add([f"noise{i}"], _unit())
    result = index.search(v[0], top_k=6)
    # target should be in results with a high score
    ids = [r[0] for r in result]
    assert "target" in ids


def test_search_returns_top_k_limit(index):
    for i in range(20):
        index.add([f"c{i}"], _unit())
    results = index.search(_unit()[0], top_k=5)
    assert len(results) <= 5


def test_search_empty_index_returns_empty(index):
    assert index.search(_unit()[0], top_k=10) == []


def test_remove_deleted_not_returned(index):
    v = _unit()
    index.add(["keep"], _unit())
    index.add(["remove-me"], v)
    index.remove(["remove-me"])
    results = index.search(v[0], top_k=10)
    assert all(cid != "remove-me" for cid, _ in results)


def test_len_excludes_deleted(index):
    for i in range(5):
        index.add([f"c{i}"], _unit())
    index.remove(["c0", "c1"])
    assert len(index) == 3


def test_scores_are_floats(index):
    for i in range(3):
        index.add([f"c{i}"], _unit())
    results = index.search(_unit()[0], top_k=3)
    for _, score in results:
        assert isinstance(score, float)
