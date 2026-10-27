"""Extended tests for temporal-pulse app/anomaly.py."""

from __future__ import annotations

import numpy as np

import app.anomaly as anomaly_mod
from app.anomaly import build_index, find_similar_anomalies


class TestBuildIndex:
    def setup_method(self) -> None:
        anomaly_mod._INDEX = None
        anomaly_mod._NN_MODEL = None
        anomaly_mod._INDEX_LABELS = []
        anomaly_mod._FEATURE_NAMES = []

    def test_build_sets_labels(self) -> None:
        X = np.random.rand(10, 4).astype(np.float32)
        labels = [{"id": i} for i in range(10)]
        build_index(X, labels, ["a", "b", "c", "d"])
        assert len(anomaly_mod._INDEX_LABELS) == 10

    def test_build_sets_feature_names(self) -> None:
        X = np.random.rand(5, 3).astype(np.float32)
        labels = [{"id": i} for i in range(5)]
        build_index(X, labels, ["x", "y", "z"])
        assert anomaly_mod._FEATURE_NAMES == ["x", "y", "z"]

    def test_build_with_no_faiss(self) -> None:
        orig = anomaly_mod.FAISS_AVAILABLE
        anomaly_mod.FAISS_AVAILABLE = False
        X = np.random.rand(5, 3).astype(np.float32)
        labels = [{"id": i} for i in range(5)]
        try:
            build_index(X, labels, ["x", "y", "z"])
            assert anomaly_mod._NN_MODEL is not None
        finally:
            anomaly_mod.FAISS_AVAILABLE = orig


class TestFindSimilarAnomalies:
    def setup_method(self) -> None:
        anomaly_mod._INDEX = None
        anomaly_mod._NN_MODEL = None
        anomaly_mod._INDEX_LABELS = []
        anomaly_mod._FEATURE_NAMES = []

    def test_empty_index_returns_empty(self) -> None:
        query = np.random.rand(4).astype(np.float32)
        result = find_similar_anomalies(query, k=3)
        assert result == []

    def test_returns_list_after_build(self) -> None:
        orig = anomaly_mod.FAISS_AVAILABLE
        anomaly_mod.FAISS_AVAILABLE = False
        X = np.random.rand(10, 4).astype(np.float32)
        labels = [{"sensor": f"s{i}"} for i in range(10)]
        build_index(X, labels, ["a", "b", "c", "d"])
        query = np.random.rand(4).astype(np.float32)
        result = find_similar_anomalies(query, k=3)
        assert isinstance(result, list)
        anomaly_mod.FAISS_AVAILABLE = orig
