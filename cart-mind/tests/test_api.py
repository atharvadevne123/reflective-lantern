"""Tests for Cart-Mind FastAPI endpoints."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client) -> None:
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_schema(self, client) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "index_loaded" in data
        assert "version" in data

    def test_health_status_value(self, client) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client) -> None:
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200

    def test_metrics_fields(self, client) -> None:
        data = client.get("/api/v1/metrics").json()
        assert "total_requests" in data
        assert "uptime_seconds" in data
        assert "p95_latency_ms" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client, intent_payload) -> None:
        r = client.post("/api/v1/predict", json=intent_payload)
        assert r.status_code == 200

    def test_predict_probability_range(self, client, intent_payload) -> None:
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert 0.0 <= data["purchase_probability"] <= 1.0

    def test_predict_returns_boolean_label(self, client, intent_payload) -> None:
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert isinstance(data["will_purchase"], bool)

    def test_predict_confidence_values(self, client, intent_payload) -> None:
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert data["confidence"] in ("high", "medium")

    def test_predict_correlation_id_returned(self, client, intent_payload) -> None:
        data = client.post("/api/v1/predict", json=intent_payload).json()
        assert "correlation_id" in data
        assert len(data["correlation_id"]) > 0

    @pytest.mark.parametrize("age", [0, 18, 65, 120])
    def test_predict_edge_ages(self, client, intent_payload, age) -> None:
        payload = {**intent_payload, "user_age": age}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200

    def test_predict_invalid_age_rejected(self, client, intent_payload) -> None:
        payload = {**intent_payload, "user_age": 150}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 422

    def test_predict_missing_user_id_rejected(self, client, intent_payload) -> None:
        payload = {k: v for k, v in intent_payload.items() if k != "user_id"}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 422


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client, recommend_payload) -> None:
        r = client.post("/api/v1/recommend", json=recommend_payload)
        assert r.status_code == 200

    def test_recommend_count_matches_top_k(self, client, recommend_payload) -> None:
        data = client.post("/api/v1/recommend", json=recommend_payload).json()
        assert data["count"] == len(data["recommendations"])

    def test_recommend_scores_in_range(self, client, recommend_payload) -> None:
        data = client.post("/api/v1/recommend", json=recommend_payload).json()
        for rec in data["recommendations"]:
            assert 0.0 <= rec["score"] <= 1.0

    def test_recommend_top_k_respected(self, client, recommend_payload) -> None:
        payload = {**recommend_payload, "top_k": 3}
        data = client.post("/api/v1/recommend", json=payload).json()
        assert data["count"] == 3

    def test_recommend_has_rank_field(self, client, recommend_payload) -> None:
        data = client.post("/api/v1/recommend", json=recommend_payload).json()
        for i, rec in enumerate(data["recommendations"], 1):
            assert rec["rank"] == i


class TestSimilarEndpoint:
    def test_similar_returns_200(self, client, similar_payload) -> None:
        r = client.post("/api/v1/similar", json=similar_payload)
        assert r.status_code == 200

    def test_similar_count_matches_top_k(self, client, similar_payload) -> None:
        data = client.post("/api/v1/similar", json=similar_payload).json()
        assert data["count"] == len(data["similar_items"])

    def test_similar_similarity_scores_positive(self, client, similar_payload) -> None:
        data = client.post("/api/v1/similar", json=similar_payload).json()
        for item in data["similar_items"]:
            assert item["similarity_score"] >= 0.0

    def test_similar_seed_id_echoed(self, client, similar_payload) -> None:
        data = client.post("/api/v1/similar", json=similar_payload).json()
        assert data["seed_item_id"] == similar_payload["item_id"]


class TestDriftEndpoint:
    def test_drift_returns_200(self, client) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        payload = {
            "feature_values": {
                "item_price": rng.uniform(5, 500, 50).tolist(),
                "avg_order_value": rng.uniform(10, 300, 50).tolist(),
            }
        }
        r = client.post("/api/v1/drift", json=payload)
        assert r.status_code == 200

    def test_drift_result_keys(self, client) -> None:
        import numpy as np

        rng = np.random.default_rng(1)
        payload = {"feature_values": {"item_price": rng.uniform(5, 500, 50).tolist()}}
        data = client.post("/api/v1/drift", json=payload).json()
        assert "results" in data
        assert "drifted_features" in data
        assert "total_checked" in data

    def test_drift_empty_feature_dict(self, client) -> None:
        payload = {"feature_values": {}}
        r = client.post("/api/v1/drift", json=payload)
        assert r.status_code == 200

    import pytest

    @pytest.mark.parametrize("n_samples", [10, 50, 100])
    def test_drift_various_sample_sizes(self, client, n_samples: int) -> None:
        import numpy as np

        rng = np.random.default_rng(42)
        payload = {"feature_values": {"item_price": rng.uniform(5, 500, n_samples).tolist()}}
        r = client.post("/api/v1/drift", json=payload)
        assert r.status_code == 200

    def test_drift_total_checked_matches_features(self, client) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        payload = {
            "feature_values": {
                "item_price": rng.uniform(5, 500, 50).tolist(),
                "avg_order_value": rng.uniform(10, 300, 50).tolist(),
            }
        }
        data = client.post("/api/v1/drift", json=payload).json()
        assert data["total_checked"] == 2
