"""Tests for forge-guard app/faiss_index.py."""

from __future__ import annotations

import numpy as np
import pytest

from app.faiss_index import build_index, is_anomalous, load_index


class TestBuildIndex:
    def test_build_does_not_raise(self) -> None:
        vectors = np.random.rand(10, 4).astype(np.float32)
        build_index(vectors, save=False)

    def test_build_with_single_vector(self) -> None:
        v = np.random.rand(1, 4).astype(np.float32)
        build_index(v, save=False)


class TestIsAnomalous:
    def test_returns_dict(self) -> None:
        result = is_anomalous(np.random.rand(4).astype(np.float32))
        assert isinstance(result, dict)

    def test_has_anomalous_key(self) -> None:
        result = is_anomalous(np.random.rand(4).astype(np.float32))
        assert "anomalous" in result

    def test_has_nn_distance_key(self) -> None:
        result = is_anomalous(np.random.rand(4).astype(np.float32))
        assert "nn_distance" in result

    def test_no_index_not_anomalous(self) -> None:
        import app.faiss_index as fi
        original = fi._index
        fi._index = None
        result = is_anomalous(np.random.rand(4).astype(np.float32))
        fi._index = original
        assert result["anomalous"] is False

    def test_anomalous_after_build(self) -> None:
        vectors = np.zeros((5, 4), dtype=np.float32)
        build_index(vectors, save=False)
        query = np.ones(4, dtype=np.float32) * 1000.0
        result = is_anomalous(query, threshold=1.0)
        assert result["anomalous"] is True


class TestLoadIndex:
    def test_returns_bool(self) -> None:
        result = load_index()
        assert isinstance(result, bool)
