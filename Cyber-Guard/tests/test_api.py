"""API endpoint tests for Cyber-Guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["version"] == "1.0.0"


def test_health_reports_all_dependencies(client: TestClient):
    """Readiness must cover the database, not just that the process is up."""
    body = client.get("/api/v1/health").json()
    assert body["model_loaded"] is True
    assert body["anomaly_model_loaded"] is True
    assert body["database_reachable"] is True


def test_health_degrades_when_database_down(client: TestClient, monkeypatch):
    """A dead database must surface as degraded, not healthy."""
    from sqlalchemy.exc import OperationalError

    from app import main

    class _DeadSession:
        def execute(self, *a, **k):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    main.app.dependency_overrides[main.get_db] = lambda: _DeadSession()
    try:
        body = client.get("/api/v1/health").json()
    finally:
        main.app.dependency_overrides.pop(main.get_db, None)

    assert body["status"] == "degraded"
    assert body["database_reachable"] is False


def test_predict_valid_payload(client: TestClient, sample_request_payload: dict):
    resp = client.post("/api/v1/predict", json=sample_request_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert "class_probabilities" in body
    assert isinstance(body["class_probabilities"], dict)


def test_predict_unknown_protocol(client: TestClient, sample_request_payload: dict):
    payload = {**sample_request_payload, "protocol_type": "unknown_xyz"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_bytes_rejected(client: TestClient, sample_request_payload: dict):
    payload = {**sample_request_payload, "src_bytes": -1}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize("protocol", ["tcp", "udp", "icmp"])
def test_predict_all_protocols(client: TestClient, sample_request_payload: dict, protocol: str):
    payload = {**sample_request_payload, "protocol_type": protocol}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_metrics_endpoint(client: TestClient):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_predictions" in body
    assert "hours" in body
    assert "class_counts" in body


def test_drift_endpoint(client: TestClient):
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert "ks_statistic" in body
    assert "p_value" in body
    assert "drift_detected" in body


def test_correlation_id_header_forwarded(client: TestClient, sample_request_payload: dict):
    cid = "test-correlation-id-123"
    resp = client.post("/api/v1/predict", json=sample_request_payload, headers={"X-Correlation-ID": cid})
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == cid


def test_response_time_header_present(client: TestClient):
    resp = client.get("/api/v1/health")
    assert "X-Response-Time-Ms" in resp.headers
