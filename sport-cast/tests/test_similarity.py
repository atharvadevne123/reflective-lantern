"""FAISS similarity search tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.similarity import _BruteForceIndex, build_index, search_similar_matches


@pytest.fixture()
def sample_vectors():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (50, 10)).astype(np.float32)


def test_brute_force_returns_top_k(sample_vectors):
    idx = _BruteForceIndex(sample_vectors)
    query = sample_vectors[0:1]
    results = idx.search(query, top_k=3)
    assert len(results) == 3
    assert results[0]["rank"] == 1


def test_brute_force_nearest_is_self(sample_vectors):
    idx = _BruteForceIndex(sample_vectors)
    query = sample_vectors[5:6]
    results = idx.search(query, top_k=1)
    assert results[0]["index"] == 5
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-4)


def test_build_index_no_persist(sample_vectors):
    idx = build_index(sample_vectors, persist=False)
    assert idx is not None


def test_search_similar_matches_empty_when_no_index():
    results = search_similar_matches(np.zeros(10, dtype=np.float32), index=None)
    assert results == []


def test_search_similar_matches_with_brute_force(sample_vectors):
    idx = _BruteForceIndex(sample_vectors)
    results = search_similar_matches(sample_vectors[0], top_k=5, index=idx)
    assert len(results) <= 5
    assert all("distance" in r for r in results)
