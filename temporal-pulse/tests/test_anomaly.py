"""Tests for FAISS/sklearn anomaly root cause analysis."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture()
def built_index():
    """Build a small anomaly index."""
    from app import anomaly

    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (50, 8)).astype(np.float32)
    labels = [{"event_id": i, "cause": f"cause-{i % 3}"} for i in range(50)]
    feature_names = [f"feat_{i}" for i in range(8)]
    anomaly.build_index(X, labels, feature_names)
    yield anomaly
    anomaly._INDEX = None
    anomaly._NN_MODEL = None
    anomaly._INDEX_LABELS = []


class TestBuildIndex:
    def test_index_is_queryable(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=3)
        assert len(results) == 3

    def test_results_have_distance(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=2)
        for r in results:
            assert "distance" in r
            assert r["distance"] >= 0.0

    def test_results_preserve_labels(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=1)
        assert "cause" in results[0]

    def test_empty_index_returns_empty(self):
        from app import anomaly

        anomaly._INDEX = None
        anomaly._NN_MODEL = None
        result = anomaly.find_similar_anomalies(np.zeros(8, dtype=np.float32))
        assert result == []


class TestExplainAnomaly:
    def test_explain_returns_contributors(self, built_index):
        query = np.array([0, 0, 0, 10.0, 0, 0, 0, 0], dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        assert "top_contributors" in explanation

    def test_extreme_feature_ranked_first(self, built_index):
        query = np.array([0, 0, 0, 100.0, 0, 0, 0, 0], dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        top = explanation["top_contributors"][0]
        assert top["feature"] == "feat_3"

    def test_explain_includes_similar_anomalies(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        assert "similar_historical_anomalies" in explanation

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_top_k_respected(self, built_index, top_k):
        query = np.arange(8, dtype=np.float32)
        explanation = built_index.explain_anomaly(query, top_k=top_k)
        assert len(explanation["top_contributors"]) == top_k
