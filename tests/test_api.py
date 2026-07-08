"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_includes_model_version(client: TestClient) -> None:
    data = client.get("/health").json()
    assert "model_version" in data
    assert data["model_version"] == "1.0.0"


def test_predict_success(client: TestClient, sample_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["route_id"] == "R001"
    assert data["congestion_level"] in {0, 1, 2, 3}


def test_predict_probabilities_sum_to_one(client: TestClient, sample_payload: dict) -> None:
    data = client.post("/api/v1/predict", json=sample_payload).json()
    total = sum(data["class_probabilities"].values())
    assert abs(total - 1.0) < 0.02


def test_predict_class_labels(client: TestClient, sample_payload: dict) -> None:
    data = client.post("/api/v1/predict", json=sample_payload).json()
    assert set(data["class_probabilities"].keys()) == {"free", "moderate", "congested", "severe"}


@pytest.mark.parametrize(
    "hour",
    [8, 17, 3, 14],
    ids=["morning-peak", "evening-peak", "night", "midday"],
)
def test_predict_different_hours(client: TestClient, sample_payload: dict, hour: int) -> None:
    resp = client.post("/api/v1/predict", json={**sample_payload, "hour": hour})
    assert resp.status_code == 200


def test_predict_invalid_road_type(client: TestClient, sample_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json={**sample_payload, "road_type": "dirt_track"})
    assert resp.status_code == 422


def test_predict_invalid_hour_above_max(client: TestClient, sample_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json={**sample_payload, "hour": 25})
    assert resp.status_code == 422


def test_predict_invalid_vehicle_count(client: TestClient, sample_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json={**sample_payload, "vehicle_count": -1})
    assert resp.status_code == 422


def test_metrics_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "congestion_distribution" in data


def test_drift_detects_shift(
    client: TestClient,
    reference_distribution: list[float],
    drifted_distribution: list[float],
) -> None:
    resp = client.post(
        "/api/v1/drift",
        json={
            "feature_name": "vehicle_count",
            "reference": reference_distribution,
            "current": drifted_distribution,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["drift_detected"] is True


def test_correlation_id_in_response_headers(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-correlation-id" in resp.headers


def test_predict_highway_road(client: TestClient, sample_payload: dict) -> None:
    resp = client.post(
        "/api/v1/predict",
        json={
            **sample_payload,
            "road_type": "highway",
            "vehicle_count": 3000,
            "avg_speed_kmh": 110.0,
        },
    )
    assert resp.status_code == 200


def test_batch_predict_success(client: TestClient, sample_payload: dict) -> None:
    batch = [sample_payload, {**sample_payload, "route_id": "R002", "hour": 17}]
    resp = client.post("/api/v1/predict/batch", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[1]["route_id"] == "R002"


def test_batch_predict_over_limit_rejected(client: TestClient, sample_payload: dict) -> None:
    batch = [{**sample_payload, "route_id": f"R{i}"} for i in range(101)]
    resp = client.post("/api/v1/predict/batch", json=batch)
    assert resp.status_code == 422


def test_model_info_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_features"] == 26
    assert "hour_sin" in data["features"]


def test_predict_missing_required_field(client: TestClient, sample_payload: dict) -> None:
    payload = {k: v for k, v in sample_payload.items() if k != "route_id"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_route_id_too_long(client: TestClient, sample_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json={**sample_payload, "route_id": "X" * 65})
    assert resp.status_code == 422


def test_route_history_endpoint(client: TestClient, sample_payload: dict) -> None:
    client.post("/api/v1/predict", json={**sample_payload, "route_id": "HIST-1"})
    resp = client.get("/api/v1/routes/HIST-1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["route_id"] == "HIST-1"
    assert data["count"] >= 1


def test_route_history_empty_for_unknown_route(client: TestClient) -> None:
    resp = client.get("/api/v1/routes/NO-SUCH-ROUTE/history")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_route_history_limit_clamped(client: TestClient) -> None:
    resp = client.get("/api/v1/routes/HIST-1/history?limit=9999")
    assert resp.status_code == 200
    assert resp.json()["count"] <= 200
