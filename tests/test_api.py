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


def test_readiness_probe_before_train(client: TestClient) -> None:
    """Readiness probe should return 503 when models are not loaded."""
    import app.main as main_mod

    orig_model = main_mod._model_bundle
    orig_anomaly = main_mod._anomaly_bundle
    main_mod._model_bundle = None
    main_mod._anomaly_bundle = None
    try:
        r = client.get("/api/v1/ready")
        assert r.status_code == 503
    finally:
        main_mod._model_bundle = orig_model
        main_mod._anomaly_bundle = orig_anomaly


def test_readiness_probe_after_train(client: TestClient) -> None:
    """Readiness probe should return 200 after models are loaded."""
    client.post("/api/v1/train")
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_version_endpoint(client: TestClient) -> None:
    """Version endpoint should return a version string."""
    r = client.get("/api/v1/version")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert data["version"] != ""


def test_regions_endpoint(client: TestClient) -> None:
    """Regions endpoint should return a list of grid regions."""
    r = client.get("/api/v1/regions")
    assert r.status_code == 200
    regions = r.json()
    assert isinstance(regions, list)
    assert len(regions) > 0
    assert all("id" in region for region in regions)


def test_drift_status_endpoint(client: TestClient) -> None:
    """Drift status endpoint should return summary and total predictions."""
    r = client.get("/api/v1/drift-status")
    assert r.status_code == 200
    data = r.json()
    assert "drift_reports" in data
    assert "total_predictions" in data


@pytest.mark.parametrize("top_n", [5, 10, 20])
def test_feature_importance_top_n(client: TestClient, top_n: int) -> None:
    """Feature importance endpoint should respect top_n parameter."""
    client.post("/api/v1/train")
    r = client.get("/api/v1/feature-importance", params={"top_n": top_n})
    assert r.status_code == 200
    data = r.json()
    assert len(data["features"]) <= top_n
    assert data["top_n"] == top_n


def test_validate_batch_valid_readings(client: TestClient) -> None:
    """Batch validation should return all valid for correct readings."""
    readings = [
        {
            "hour": 10,
            "day_of_week": 1,
            "month": 6,
            "temperature_c": 22.0,
            "humidity_pct": 55.0,
            "consumption_kwh": 12.5,
        }
    ]
    r = client.post("/api/v1/validate-batch", json={"readings": readings})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert data["valid_count"] == 1
    assert data["invalid_count"] == 0


def test_validate_batch_invalid_readings(client: TestClient) -> None:
    """Batch validation should catch invalid hour values."""
    readings = [
        {
            "hour": 99,
            "day_of_week": 1,
            "month": 6,
            "temperature_c": 22.0,
            "humidity_pct": 55.0,
            "consumption_kwh": 12.5,
        }
    ]
    r = client.post("/api/v1/validate-batch", json={"readings": readings})
    assert r.status_code == 200
    data = r.json()
    assert data["invalid_count"] == 1
    assert len(data["results"][0]["errors"]) > 0


def test_validate_batch_empty_raises(client: TestClient) -> None:
    """Batch validation should return 400 for empty readings."""
    r = client.post("/api/v1/validate-batch", json={"readings": []})
    assert r.status_code == 400


def test_config_endpoint(client: TestClient) -> None:
    """Config endpoint should return non-sensitive settings."""
    r = client.get("/api/v1/config")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "rate_limit_per_minute" in data
    assert "log_level" in data
    assert "database_url" not in data  # sensitive - should not be exposed directly


def test_health_has_version(client: TestClient) -> None:
    """Health endpoint should include version string."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert data["version"].count(".") >= 1  # basic semver check


def test_carbon_estimate_endpoint(client: TestClient) -> None:
    """Carbon estimate endpoint should return CO2 values."""
    r = client.get("/api/v1/carbon/estimate", params={"kwh": 100.0, "region": "default"})
    assert r.status_code == 200
    data = r.json()
    assert "co2_kg" in data
    assert "co2_tonnes" in data
    assert "trees_equivalent" in data
    assert data["co2_kg"] > 0


def test_carbon_estimate_negative_kwh(client: TestClient) -> None:
    """Carbon estimate should return 422 for negative kWh."""
    r = client.get("/api/v1/carbon/estimate", params={"kwh": -10.0})
    assert r.status_code == 422


def test_carbon_annual_report_endpoint(client: TestClient) -> None:
    """Annual carbon report should accept 12 monthly values."""
    monthly = [100.0] * 12
    r = client.post("/api/v1/carbon/annual-report", params={"region": "northeast"}, json=monthly)
    assert r.status_code == 200
    data = r.json()
    assert "total_co2_kg" in data
    assert len(data["monthly_co2_kg"]) == 12


def test_carbon_annual_report_wrong_months(client: TestClient) -> None:
    """Annual carbon report should return 422 for non-12 month list."""
    monthly = [100.0] * 11
    r = client.post("/api/v1/carbon/annual-report", json=monthly)
    assert r.status_code == 422


def test_export_csv_endpoint(client: TestClient) -> None:
    """Export endpoint should return a CSV string and summary."""
    records = [{"hour": 8, "consumption_kwh": 12.0}, {"hour": 14, "consumption_kwh": 20.0}]
    r = client.post("/api/v1/export/csv", json=records)
    assert r.status_code == 200
    data = r.json()
    assert "csv" in data
    assert "summary" in data
    assert data["summary"]["total_records"] == 2


def test_data_quality_endpoint(client: TestClient) -> None:
    """Data quality endpoint should return scored records and summary."""
    records = [
        {"hour": 10, "month": 6, "day_of_week": 2, "consumption_kwh": 15.0},
        {"hour": 99, "month": 6, "day_of_week": 2, "consumption_kwh": 15.0},
    ]
    r = client.post("/api/v1/data-quality", json=records)
    assert r.status_code == 200
    data = r.json()
    assert "scored_records" in data
    assert "summary" in data
    assert data["summary"]["total_records"] == 2


def test_data_quality_empty_raises(client: TestClient) -> None:
    """Data quality endpoint should return 400 for empty input."""
    r = client.post("/api/v1/data-quality", json=[])
    assert r.status_code == 400


def test_trend_endpoint_rising(client: TestClient) -> None:
    """Trend endpoint should detect a rising series."""
    r = client.post("/api/v1/trend", json=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert r.status_code == 200
    data = r.json()
    assert data["direction"] == "rising"
    assert data["slope"] > 0


def test_trend_endpoint_too_short(client: TestClient) -> None:
    """Trend endpoint should return 422 for single value."""
    r = client.post("/api/v1/trend", json=[5.0])
    assert r.status_code == 422


def test_cache_stats_endpoint(client: TestClient) -> None:
    """Cache stats endpoint should return hit/miss statistics."""
    r = client.get("/api/v1/cache/stats")
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data


def test_naive_forecast_endpoint(client: TestClient) -> None:
    """Naive forecast endpoint should return correct number of steps."""
    r = client.post("/api/v1/forecast/naive", json=[10.0, 12.0, 11.0, 14.0], params={"steps": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["steps"] == 5
    assert len(data["forecast"]) == 5
    assert data["method"] == "naive"


def test_naive_forecast_empty_raises(client: TestClient) -> None:
    """Naive forecast should return 422 for empty input."""
    r = client.post("/api/v1/forecast/naive", json=[], params={"steps": 3})
    assert r.status_code == 422


def test_ses_forecast_endpoint(client: TestClient) -> None:
    """SES forecast endpoint should return smoothed forecast."""
    r = client.post("/api/v1/forecast/ses", json=[10.0, 12.0, 11.0, 14.0], params={"steps": 3, "alpha": 0.4})
    assert r.status_code == 200
    data = r.json()
    assert data["steps"] == 3
    assert data["method"] == "ses"
    assert data["alpha"] == pytest.approx(0.4)


def test_ses_forecast_invalid_alpha(client: TestClient) -> None:
    """SES forecast should return 422 for invalid alpha."""
    r = client.post("/api/v1/forecast/ses", json=[10.0, 12.0], params={"steps": 3, "alpha": 0.0})
    assert r.status_code == 422


@pytest.mark.parametrize("steps", [1, 6, 24])
def test_naive_forecast_various_steps(client: TestClient, steps: int) -> None:
    """Naive forecast endpoint should respect steps parameter."""
    r = client.post("/api/v1/forecast/naive", json=[15.0] * 10, params={"steps": steps})
    assert r.status_code == 200
    assert r.json()["steps"] == steps
