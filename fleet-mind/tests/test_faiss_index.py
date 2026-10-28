"""Tests for FAISS nearest-neighbour index (brute-force fallback path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.faiss_index as fi


@pytest.fixture(autouse=True)
def reset_index():
    fi._index = None
    fi._stored_ids = []
    yield
    fi._index = None
    fi._stored_ids = []


@pytest.fixture
def sample_vectors():
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 18)).astype(np.float32)


@pytest.fixture
def sample_ids():
    return [f"VH-{i:03d}" for i in range(20)]


class TestBuildIndex:
    def test_builds_without_error(self, sample_vectors, sample_ids):
        fi.build_index(sample_vectors, sample_ids)
        assert fi._index is not None

    def test_stored_ids_match(self, sample_vectors, sample_ids):
        fi.build_index(sample_vectors, sample_ids)
        assert fi._stored_ids == sample_ids


class TestQuerySimilar:
    def test_no_index_returns_empty(self):
        q = np.zeros((1, 18), dtype=np.float32)
        assert fi.query_similar(q) == []

    def test_returns_k_results(self, sample_vectors, sample_ids):
        fi.build_index(sample_vectors, sample_ids)
        results = fi.query_similar(sample_vectors[0], k=3)
        assert len(results) == 3

    def test_result_keys(self, sample_vectors, sample_ids):
        fi.build_index(sample_vectors, sample_ids)
        results = fi.query_similar(sample_vectors[0], k=1)
        assert "vehicle_id" in results[0]
        assert "distance" in results[0]

    def test_nearest_neighbour_is_itself(self, sample_vectors, sample_ids):
        fi.build_index(sample_vectors, sample_ids)
        results = fi.query_similar(sample_vectors[0], k=1)
        assert results[0]["vehicle_id"] == sample_ids[0]

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_parametrized_k(self, sample_vectors, sample_ids, k):
        fi.build_index(sample_vectors, sample_ids)
        results = fi.query_similar(sample_vectors[0], k=k)
        assert 1 <= len(results) <= k
