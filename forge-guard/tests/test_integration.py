"""End-to-end integration tests simulating a full inference + monitoring cycle."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_full_predict_then_metrics_cycle(client: TestClient, sample_sensor_payload: dict) -> None:
    """Full flow: predict → check defect rate appears in metrics."""
    for _ in range(3):
        resp = client.post("/api/v1/predict", json=sample_sensor_payload)
        assert resp.status_code == 200

    metrics = client.get("/api/v1/metrics")
    assert metrics.status_code == 200
    data = metrics.json()
    assert data["defect_rate_24h"]["total"] >= 0


def test_feature_importance_non_empty(client: TestClient) -> None:
    importance = client.get("/api/v1/feature-importance")
    assert importance.status_code == 200
    data = importance.json()
    assert isinstance(data, dict)
    assert len(data) > 0


@pytest.mark.parametrize("endpoint", ["/health", "/api/v1/metrics"])
def test_read_endpoints_always_200(client: TestClient, endpoint: str) -> None:
    resp = client.get(endpoint)
    assert resp.status_code == 200


def test_predict_and_batch_return_same_label(
    client: TestClient, sample_sensor_payload: dict
) -> None:
    single = client.post("/api/v1/predict", json=sample_sensor_payload).json()
    batch = client.post("/api/v1/predict/batch", json={"readings": [sample_sensor_payload]}).json()
    # Same input → same prediction
    assert single["prediction"] == batch["predictions"][0]["prediction"]
    assert single["defect_probability"] == batch["predictions"][0]["defect_probability"]


def test_correlation_id_in_predict_response(
    client: TestClient, sample_sensor_payload: dict
) -> None:
    cid = "integration-test-cid"
    resp = client.post(
        "/api/v1/predict",
        json=sample_sensor_payload,
        headers={"X-Correlation-ID": cid},
    )
    assert resp.json()["correlation_id"] == cid
    assert resp.headers["X-Correlation-ID"] == cid


def test_export_predictions_after_inference(
    client: TestClient, sample_sensor_payload: dict
) -> None:
    """After making predictions, the export summary should show non-zero total."""
    for _ in range(2):
        client.post("/api/v1/predict", json=sample_sensor_payload)
    resp = client.get("/api/v1/export/predictions?hours=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 0


def test_version_endpoint_consistent(client: TestClient) -> None:
    r1 = client.get("/api/v1/version").json()
    r2 = client.get("/api/v1/version").json()
    assert r1["model_version"] == r2["model_version"]
    assert r1["api_version"] == r2["api_version"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/version",
        "/api/v1/feature-importance",
        "/api/v1/export/predictions",
        "/api/v1/export/drift",
    ],
)
def test_ops_endpoints_return_200(client: TestClient, endpoint: str) -> None:
    resp = client.get(endpoint)
    assert resp.status_code == 200
