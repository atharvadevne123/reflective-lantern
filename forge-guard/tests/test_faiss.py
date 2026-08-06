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
