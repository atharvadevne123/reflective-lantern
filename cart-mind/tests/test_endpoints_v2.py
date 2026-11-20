"""Tests for batch prediction, model info, and cache stats endpoints."""

from __future__ import annotations

import pytest


class TestBatchPredict:
    def test_batch_returns_200(self, client, intent_payload) -> None:
        payload = {"items": [intent_payload, intent_payload]}
        r = client.post("/api/v1/predict/batch", json=payload)
        assert r.status_code == 200

    def test_batch_count_matches_input(self, client, intent_payload) -> None:
        payload = {"items": [intent_payload] * 4}
        data = client.post("/api/v1/predict/batch", json=payload).json()
        assert data["count"] == 4
        assert len(data["predictions"]) == 4

    def test_batch_probabilities_in_range(self, client, intent_payload) -> None:
        payload = {"items": [intent_payload] * 3}
        data = client.post("/api/v1/predict/batch", json=payload).json()
        for pred in data["predictions"]:
            assert 0.0 <= pred["purchase_probability"] <= 1.0

    def test_batch_mean_probability_in_range(self, client, intent_payload) -> None:
        payload = {"items": [intent_payload] * 3}
        data = client.post("/api/v1/predict/batch", json=payload).json()
        assert 0.0 <= data["mean_probability"] <= 1.0

    def test_batch_echoes_ids(self, client, intent_payload) -> None:
        payload = {"items": [intent_payload]}
        data = client.post("/api/v1/predict/batch", json=payload).json()
        assert data["predictions"][0]["user_id"] == intent_payload["user_id"]
        assert data["predictions"][0]["item_id"] == intent_payload["item_id"]

    def test_empty_batch_rejected(self, client) -> None:
        r = client.post("/api/v1/predict/batch", json={"items": []})
        assert r.status_code == 422

    def test_oversized_batch_rejected(self, client, intent_payload) -> None:
        r = client.post("/api/v1/predict/batch", json={"items": [intent_payload] * 501})
        assert r.status_code == 422

    def test_batch_matches_single_prediction(self, client, intent_payload) -> None:
        """A row scored in a batch must get the same probability as scored alone."""
        single = client.post("/api/v1/predict", json=intent_payload).json()
        batch = client.post("/api/v1/predict/batch", json={"items": [intent_payload]}).json()
        assert single["purchase_probability"] == pytest.approx(
            batch["predictions"][0]["purchase_probability"], abs=1e-4
        )

    @pytest.mark.parametrize("n", [1, 2, 10, 50])
    def test_various_batch_sizes(self, client, intent_payload, n) -> None:
        data = client.post("/api/v1/predict/batch", json={"items": [intent_payload] * n}).json()
        assert data["count"] == n


class TestModelInfo:
    def test_model_info_returns_200(self, client) -> None:
        assert client.get("/api/v1/model/info").status_code == 200

    def test_reports_ensemble_members(self, client) -> None:
        data = client.get("/api/v1/model/info").json()
        assert set(data["ensemble_members"]) == {"LightGBM", "XGBoost", "RandomForest"}

    def test_reports_five_feature_stages(self, client) -> None:
        data = client.get("/api/v1/model/info").json()
        assert len(data["feature_stages"]) == 5

    def test_model_version_present(self, client) -> None:
        data = client.get("/api/v1/model/info").json()
        assert data["model_version"]

    def test_auc_is_null_or_valid(self, client) -> None:
        data = client.get("/api/v1/model/info").json()
        if data["auc_mean"] is not None:
            assert 0.0 <= data["auc_mean"] <= 1.0


class TestCacheStats:
    def test_cache_stats_returns_200(self, client) -> None:
        assert client.get("/api/v1/cache/stats").status_code == 200

    def test_both_caches_reported(self, client) -> None:
        data = client.get("/api/v1/cache/stats").json()
        assert "recommendation_cache" in data
        assert "similarity_cache" in data

    def test_stats_include_hit_rate(self, client) -> None:
        data = client.get("/api/v1/cache/stats").json()
        assert "hit_rate" in data["recommendation_cache"]
        assert "entries" in data["similarity_cache"]

    def test_repeated_similar_call_registers_hit(self, client, similar_payload) -> None:
        """The second identical similarity request must be served from cache."""
        payload = {**similar_payload, "item_id": "cache_probe_item"}
        client.post("/api/v1/similar", json=payload)
        before = client.get("/api/v1/cache/stats").json()["similarity_cache"]["hits"]
        client.post("/api/v1/similar", json=payload)
        after = client.get("/api/v1/cache/stats").json()["similarity_cache"]["hits"]
        assert after > before

    def test_cached_similar_response_is_identical(self, client, similar_payload) -> None:
        payload = {**similar_payload, "item_id": "cache_identity_item"}
        first = client.post("/api/v1/similar", json=payload).json()
        second = client.post("/api/v1/similar", json=payload).json()
        assert first["similar_items"] == second["similar_items"]
