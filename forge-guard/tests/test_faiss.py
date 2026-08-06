"""Tests for FAISS anomaly detection module."""

from __future__ import annotations

import numpy as np
import pytest


def test_is_anomalous_no_index_returns_false():
    from app import faiss_index

    faiss_index._index = None
    result = faiss_index.is_anomalous(np.zeros((1, 10)))
    assert result["anomalous"] is False
    assert result["nn_distance"] is None


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faiss"),
    reason="faiss-cpu not installed",
)
def test_build_and_query_index(tmp_path, monkeypatch):
    from app import faiss_index

    monkeypatch.setattr(faiss_index, "INDEX_PATH", tmp_path / "idx.bin")
    monkeypatch.setattr(faiss_index, "REF_VECTORS_PATH", tmp_path / "ref.npy")

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 20)).astype(np.float32)
    faiss_index.build_index(ref, save=True)

    # A near-zero query should be close to the reference cluster
    query = rng.normal(0, 0.5, (1, 20)).astype(np.float32)
    result = faiss_index.is_anomalous(query, k=5, threshold=1000.0)
    assert result["anomalous"] is False
    assert result["nn_distance"] is not None

    # A very distant query should be flagged anomalous
    far_query = rng.normal(500, 1, (1, 20)).astype(np.float32)
    far_result = faiss_index.is_anomalous(far_query, k=5, threshold=1.0)
    assert far_result["anomalous"] is True


def test_is_anomalous_result_keys():
    from app import faiss_index

    faiss_index._index = None
    result = faiss_index.is_anomalous(np.zeros((1, 10)))
    assert "anomalous" in result
    assert "nn_distance" in result
    assert "nn_k" in result


def test_load_index_missing_files_returns_false(tmp_path, monkeypatch):
    from app import faiss_index

    monkeypatch.setattr(faiss_index, "INDEX_PATH", tmp_path / "nonexistent.bin")
    monkeypatch.setattr(faiss_index, "REF_VECTORS_PATH", tmp_path / "nonexistent.npy")
    faiss_index._index = None
    result = faiss_index.load_index()
    assert result is False


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faiss"),
    reason="faiss-cpu not installed",
)
def test_is_anomalous_k_parameter(tmp_path, monkeypatch, k):
    from app import faiss_index

    monkeypatch.setattr(faiss_index, "INDEX_PATH", tmp_path / "idx.bin")
    monkeypatch.setattr(faiss_index, "REF_VECTORS_PATH", tmp_path / "ref.npy")

    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, (100, 10)).astype(np.float32)
    faiss_index.build_index(ref, save=True)

    query = rng.normal(0, 0.5, (1, 10)).astype(np.float32)
    result = faiss_index.is_anomalous(query, k=k, threshold=1000.0)
    assert result["nn_k"] == k
