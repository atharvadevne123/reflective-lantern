"""Tests for FAISS-based attack pattern retrieval."""
from __future__ import annotations

import numpy as np
import pytest


def test_get_index_returns_instance() -> None:
    from app.retrieval import get_index

    idx = get_index()
    assert idx is not None


def test_index_has_seeded_patterns() -> None:
    from app.retrieval import AttackPatternIndex

    idx = AttackPatternIndex(dim=25)
    assert idx.size > 0


def test_search_returns_list() -> None:
    from app.retrieval import get_index

    idx = get_index()
    results = idx.search([0.0] * 25, k=3)
    assert isinstance(results, list)


def test_search_result_has_rank_and_similarity() -> None:
    from app.retrieval import get_index

    idx = get_index()
    results = idx.search([1.0] * 25, k=1)
    if results:
        assert "rank" in results[0]
        assert "similarity" in results[0]
        assert "attack_type" in results[0]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_search_respects_k(k: int) -> None:
    from app.retrieval import get_index

    idx = get_index()
    results = idx.search([0.5] * 25, k=k)
    assert len(results) <= k


def test_search_with_numpy_array() -> None:
    from app.retrieval import get_index

    idx = get_index()
    query = np.zeros(25)
    results = idx.search(query, k=3)
    assert isinstance(results, list)


def test_attack_type_is_valid() -> None:
    from app.retrieval import get_index

    idx = get_index()
    results = idx.search([0.0] * 25, k=5)
    valid_types = {"dos", "probe", "r2l", "u2r"}
    for r in results:
        assert r["attack_type"] in valid_types


def test_singleton_pattern() -> None:
    from app.retrieval import _index_cache, get_index

    _index_cache.clear()
    idx1 = get_index()
    idx2 = get_index()
    assert idx1 is idx2


def test_clear_index_cache_creates_fresh_instance() -> None:
    from app.retrieval import clear_index_cache, get_index

    first = get_index()
    assert get_index() is first
    clear_index_cache()
    second = get_index()
    assert second is not first
    assert second.size == first.size
