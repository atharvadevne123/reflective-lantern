"""API endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_train_endpoint(client: TestClient):
    r = client.post("/api/v1/train")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "trained"
    assert "metrics" in data
    assert data["metrics"]["r2_mean"] > -1.0


def test_predict_after_train(client: TestClient, energy_payload):
    client.post("/api/v1/train")
    r = client.post("/api/v1/predict", json=energy_payload)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_kwh" in data
    assert data["predicted_kwh"] >= 0
    assert data["building_id"] == "bldg-001"
    assert data["latency_ms"] >= 0


def test_predict_no_model(client: TestClient, energy_payload):
    """Should return 503 when model not loaded (monkeypatched)."""
    import app.main as main_mod

    original = main_mod._model_bundle
    main_mod._model_bundle = None
    try:
        r = client.post("/api/v1/predict", json=energy_payload)
        assert r.status_code == 503
    finally:
        main_mod._model_bundle = original


def test_anomaly_after_train(client: TestClient, energy_payload):
    client.post("/api/v1/train")
    anomaly_payload = {**energy_payload, "consumption_kwh": 15.0}
    del anomaly_payload["consumption_kwh"]
    anomaly_payload["consumption_kwh"] = 15.0
    r = client.post("/api/v1/anomaly", json=anomaly_payload)
    assert r.status_code == 200
    data = r.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert data["severity"] in ("none", "warning", "critical")


def test_anomaly_no_model(client: TestClient, energy_payload):
    import app.main as main_mod

    original = main_mod._anomaly_bundle
    main_mod._anomaly_bundle = None
    try:
        r = client.post("/api/v1/anomaly", json=energy_payload)
        assert r.status_code == 503
    finally:
        main_mod._anomaly_bundle = original


def test_drift_endpoint(client: TestClient):
    client.post("/api/v1/train")
    payload = {
        "current_values": [12.0 + i * 0.1 for i in range(30)],
        "reference_values": [10.0 + i * 0.05 for i in range(30)],
    }
    r = client.post("/api/v1/drift", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "ks_statistic" in data
    assert "drift_detected" in data
    assert isinstance(data["drift_detected"], bool)


def test_drift_insufficient_reference(client: TestClient) -> None:
    payload = {
        "current_values": [1.0] * 20,
        "reference_values": [1.0] * 5,
    }
    r = client.post("/api/v1/drift", json=payload)
    assert r.status_code == 400


def test_metrics_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data
    assert "total_anomalies_flagged" in data


@pytest.mark.parametrize("building_id", ["bldg-A", "facility_99", "plant-002"])
def test_predict_various_buildings(client: TestClient, energy_payload, building_id):
    client.post("/api/v1/train")
    payload = {**energy_payload, "building_id": building_id}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["building_id"] == building_id


@pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
def test_predict_various_hours(client: TestClient, energy_payload, hour):
    client.post("/api/v1/train")
    payload = {**energy_payload, "hour": hour}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_kwh"] >= 0


def test_predict_invalid_building_id(client: TestClient, energy_payload):
    payload = {**energy_payload, "building_id": "bad id!@#"}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_hour(client: TestClient, energy_payload) -> None:
    payload = {**energy_payload, "hour": 25}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_batch_predict_after_train(client: TestClient, energy_payload) -> None:
    client.post("/api/v1/train")
    batch = [energy_payload, {**energy_payload, "building_id": "bldg-002"}]
    r = client.post("/api/v1/predict/batch", json=batch)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 2
    for res in results:
        assert res["predicted_kwh"] >= 0


def test_batch_predict_exceeds_limit(client: TestClient, energy_payload) -> None:
    client.post("/api/v1/train")
    batch = [energy_payload] * 101
    r = client.post("/api/v1/predict/batch", json=batch)
    assert r.status_code == 400


def test_drift_strong_shift_detected(client: TestClient) -> None:
    client.post("/api/v1/train")
    payload = {
        "current_values": [50.0 + i for i in range(30)],
        "reference_values": [1.0 + i * 0.01 for i in range(30)],
    }
    r = client.post("/api/v1/drift", json=payload)
    assert r.status_code == 200
    assert r.json()["drift_detected"] is True


def test_feature_importance_no_model(client: TestClient):
    import app.main as main_mod

    original = main_mod._model_bundle
    main_mod._model_bundle = None
    try:
        r = client.get("/api/v1/feature-importance")
        assert r.status_code == 503
    finally:
        main_mod._model_bundle = original


def test_feature_importance_after_train(client: TestClient):
    client.post("/api/v1/train")
    r = client.get("/api/v1/feature-importance")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data
    assert "top_n" in data
    assert "model_version" in data
    assert isinstance(data["features"], list)


def test_anomaly_stats_endpoint(client: TestClient):
    r = client.get("/api/v1/anomaly/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_anomalies" in data
    assert "anomaly_rate" in data
    assert "mean_anomaly_score" in data


def test_savings_endpoint(client: TestClient):
    payload = {
        "actual_kwh": [8.0, 9.0, 7.5],
        "baseline_kwh": [10.0, 11.0, 9.0],
        "tariff_per_kwh": 0.15,
    }
    r = client.post("/api/v1/savings", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "total_saved_kwh" in data
    assert "total_saved_cost" in data
    assert "savings_pct" in data


def test_savings_endpoint_mismatched_lengths(client: TestClient):
    payload = {
        "actual_kwh": [8.0, 9.0],
        "baseline_kwh": [10.0, 11.0, 9.0],
        "tariff_per_kwh": 0.15,
    }
    r = client.post("/api/v1/savings", json=payload)
    assert r.status_code == 422


def test_efficiency_grade_endpoint(client: TestClient):
    r = client.get("/api/v1/efficiency-grade", params={"actual_kwh": 8.0, "baseline_kwh": 10.0})
    assert r.status_code == 200
    data = r.json()
    assert "grade" in data
    assert data["grade"] in ("A+", "A", "A-", "B", "C", "D", "F")


@pytest.mark.parametrize("hour,expected_status", [(0, 200), (23, 200)])
def test_predict_boundary_hours(client: TestClient, energy_payload, hour, expected_status):
    client.post("/api/v1/train")
    payload = {**energy_payload, "hour": hour}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == expected_status
