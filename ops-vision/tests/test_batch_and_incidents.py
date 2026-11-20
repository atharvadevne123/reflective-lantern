"""Tests for the batch prediction and incident listing endpoints."""

import pytest

from app.crud import create_incident


def _incident_row(service: str = "batch-svc", is_incident: bool = True) -> dict:
    """Build a valid incident row for seeding the test database."""
    return {
        "service_name": service,
        "cpu_usage_pct": 82.0,
        "memory_usage_pct": 79.0,
        "error_rate_per_min": 44.0,
        "latency_p99_ms": 1200.0,
        "request_rate_per_sec": 60.0,
        "disk_io_util_pct": 74.0,
        "is_incident": is_incident,
        "severity": "high" if is_incident else None,
    }


class TestBatchPredict:
    """Tests for POST /api/v1/predict/batch."""

    def test_batch_returns_one_result_per_item(self, client, sample_metrics, normal_metrics):
        """The response length matches the request length."""
        resp = client.post(
            "/api/v1/predict/batch",
            json={"items": [sample_metrics, normal_metrics, sample_metrics]},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_batch_preserves_input_order(self, client, sample_metrics, normal_metrics):
        """Results come back in the same order as the submitted items."""
        a = dict(sample_metrics, service_name="svc-a")
        b = dict(normal_metrics, service_name="svc-b")
        c = dict(sample_metrics, service_name="svc-c")

        resp = client.post("/api/v1/predict/batch", json={"items": [a, b, c]})
        names = [r["service_name"] for r in resp.json()]
        assert names == ["svc-a", "svc-b", "svc-c"]

    def test_batch_matches_single_predict(self, client, sample_metrics):
        """A batch of one yields the same verdict as the single-item endpoint."""
        single = client.post("/api/v1/predict", json=sample_metrics).json()
        batched = client.post(
            "/api/v1/predict/batch", json={"items": [sample_metrics]}
        ).json()[0]
        assert batched["predicted_incident"] == single["predicted_incident"]
        assert batched["confidence"] == pytest.approx(single["confidence"], abs=1e-6)

    def test_empty_batch_rejected(self, client):
        """An empty item list is a validation error."""
        resp = client.post("/api/v1/predict/batch", json={"items": []})
        assert resp.status_code == 422

    def test_batch_validates_each_item(self, client, sample_metrics):
        """One invalid item rejects the whole batch."""
        bad = dict(sample_metrics, cpu_usage_pct=150.0)
        resp = client.post("/api/v1/predict/batch", json={"items": [sample_metrics, bad]})
        assert resp.status_code == 422

    def test_batch_confidences_in_range(self, client, sample_metrics, normal_metrics):
        """Every confidence in the batch is a valid probability."""
        resp = client.post(
            "/api/v1/predict/batch", json={"items": [sample_metrics, normal_metrics]}
        )
        for item in resp.json():
            assert 0.0 <= item["confidence"] <= 1.0


class TestListIncidents:
    """Tests for GET /api/v1/incidents."""

    def test_returns_seeded_incident(self, client, db_session):
        """A persisted incident appears in the listing."""
        create_incident(db_session, _incident_row("listed-svc"))
        resp = client.get("/api/v1/incidents")
        assert resp.status_code == 200
        assert any(r["service_name"] == "listed-svc" for r in resp.json())

    def test_filter_by_service_name(self, client, db_session):
        """The service_name filter excludes other services."""
        create_incident(db_session, _incident_row("wanted-svc"))
        create_incident(db_session, _incident_row("unwanted-svc"))
        resp = client.get("/api/v1/incidents", params={"service_name": "wanted-svc"})
        assert all(r["service_name"] == "wanted-svc" for r in resp.json())

    def test_limit_caps_result_count(self, client, db_session):
        """The limit parameter bounds how many rows come back."""
        for i in range(5):
            create_incident(db_session, _incident_row(f"lim-svc-{i}"))
        resp = client.get("/api/v1/incidents", params={"limit": 2})
        assert len(resp.json()) <= 2

    @pytest.mark.parametrize("limit,expected_status", [(1, 200), (500, 200), (0, 422), (501, 422)])
    def test_limit_bounds_validated(self, client, limit, expected_status):
        """limit must fall within 1-500."""
        resp = client.get("/api/v1/incidents", params={"limit": limit})
        assert resp.status_code == expected_status

    def test_negative_offset_rejected(self, client):
        """A negative offset is a validation error."""
        resp = client.get("/api/v1/incidents", params={"offset": -1})
        assert resp.status_code == 422
