"""Tests for FAISS-based similarity search module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.faiss_index import PostIndex, SimilarPost

DIM = 8


def _random_vec(rng: np.random.Generator | None = None) -> np.ndarray:
    r = rng or np.random.default_rng(0)
    return r.random(DIM).astype(np.float32)


class TestPostIndexAdd:
    def test_size_increments(self) -> None:
        idx = PostIndex(dim=DIM)
        idx.add(_random_vec())
        assert idx.size == 1

    def test_add_multiple(self) -> None:
        idx = PostIndex(dim=DIM)
        for _ in range(5):
            idx.add(_random_vec())
        assert idx.size == 5

    def test_wrong_dim_raises(self) -> None:
        idx = PostIndex(dim=DIM)
        with pytest.raises(ValueError):
            idx.add(np.zeros(DIM + 1, dtype=np.float32))


class TestPostIndexSearch:
    def test_empty_index_returns_empty(self) -> None:
        idx = PostIndex(dim=DIM)
        results = idx.search(_random_vec(), k=3)
        assert results == []

    def test_returns_at_most_k(self) -> None:
        idx = PostIndex(dim=DIM)
        rng = np.random.default_rng(42)
        for _ in range(10):
            idx.add(_random_vec(rng))
        results = idx.search(_random_vec(rng), k=3)
        assert len(results) <= 3

    def test_nearest_is_self(self) -> None:
        idx = PostIndex(dim=DIM)
        vec = _random_vec()
        idx.add(vec, metadata={"label": "self"})
        rng = np.random.default_rng(1)
        for _ in range(9):
            idx.add(_random_vec(rng))
        results = idx.search(vec, k=1)
        assert results[0].distance < 1e-4

    def test_result_type(self) -> None:
        idx = PostIndex(dim=DIM)
        idx.add(_random_vec())
        results = idx.search(_random_vec(), k=1)
        assert isinstance(results[0], SimilarPost)

    def test_metadata_preserved(self) -> None:
        idx = PostIndex(dim=DIM)
        meta = {"post_id": "abc123", "viral": True}
        idx.add(_random_vec(), metadata=meta)
        results = idx.search(_random_vec(), k=1)
        assert results[0].metadata == meta


class TestPostIndexPersistence:
    def test_save_and_load(self) -> None:
        idx = PostIndex(dim=DIM)
        rng = np.random.default_rng(7)
        for i in range(5):
            idx.add(_random_vec(rng), metadata={"i": i})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.idx"
            idx.save(path)
            loaded = PostIndex.load(path)

        assert loaded.size == 5
        assert loaded.dim == DIM


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_search_k_values(k: int) -> None:
    """Search should return at most k results for any k."""
    idx = PostIndex(dim=DIM)
    rng = np.random.default_rng(99)
    for _ in range(7):
        idx.add(_random_vec(rng))
    results = idx.search(_random_vec(rng), k=k)
    assert len(results) <= k
