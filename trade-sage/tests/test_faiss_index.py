"""Tests for FAISS equity pattern similarity search."""
from __future__ import annotations

import numpy as np
import pytest

import app.faiss_index as fi


@pytest.fixture(autouse=True)
def reset_index():
    """Ensure clean global index state for each test."""
    fi._index = None
    fi._stored_labels = []
    yield
    fi._index = None
    fi._stored_labels = []


def _make_vectors(n: int = 10, dim: int = 22) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    vectors = rng.random((n, dim)).astype(np.float32)
    labels = [f"TICK{i:03d}" for i in range(n)]
    return vectors, labels


def test_search_returns_empty_without_index():
    query = np.zeros(22, dtype=np.float32)
    results = fi.search_similar(query, k=3)
    assert results == []


def test_build_and_search_returns_k_results():
    if not fi._FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    vectors, labels = _make_vectors(20, 22)
    fi.build_index(vectors, labels)
    query = vectors[0]
    results = fi.search_similar(query, k=5)
    assert len(results) == 5


def test_nearest_neighbour_is_self():
    if not fi._FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    vectors, labels = _make_vectors(10, 22)
    fi.build_index(vectors, labels)
    results = fi.search_similar(vectors[3], k=1)
    assert len(results) == 1
    assert results[0][0] == labels[3]
    assert results[0][1] < 1e-5


def test_distances_non_negative():
    if not fi._FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    vectors, labels = _make_vectors(15, 22)
    fi.build_index(vectors, labels)
    results = fi.search_similar(vectors[0], k=5)
    for _, dist in results:
        assert dist >= 0.0


def test_labels_in_results():
    if not fi._FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    vectors, labels = _make_vectors(8, 22)
    fi.build_index(vectors, labels)
    results = fi.search_similar(vectors[0], k=3)
    returned_labels = {r[0] for r in results}
    assert returned_labels.issubset(set(labels))


def test_build_index_without_faiss_is_noop():
    orig = fi._FAISS_AVAILABLE
    fi._FAISS_AVAILABLE = False
    try:
        vectors, labels = _make_vectors(5, 22)
        fi.build_index(vectors, labels)
        assert fi._index is None
        assert fi._stored_labels == []
    finally:
        fi._FAISS_AVAILABLE = orig


def test_search_fallback_when_faiss_unavailable():
    orig = fi._FAISS_AVAILABLE
    fi._FAISS_AVAILABLE = False
    fi._stored_labels = []
    fi._index = None
    try:
        query = np.zeros(22, dtype=np.float32)
        results = fi.search_similar(query, k=3)
        assert results == []
    finally:
        fi._FAISS_AVAILABLE = orig
