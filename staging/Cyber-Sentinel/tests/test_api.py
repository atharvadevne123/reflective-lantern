"""API endpoint tests for Cyber-Sentinel."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_version(client: TestClient) -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_train_returns_metrics(client: TestClient) -> None:
    resp = client.post("/train")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "f1_score" in data["metrics"]


def test_predict_requires_training(client: TestClient, sample_event: dict) -> None:
    resp = client.post("/predict", json=sample_event)
    assert resp.status_code in (200, 503)


def test_predict_after_training(trained_client: TestClient, sample_event: dict) -> None:
    resp = trained_client.post("/predict", json=sample_event)
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert data["label"] in ("normal", "attack")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_dos_event(trained_client: TestClient, dos_event: dict) -> None:
    resp = trained_client.post("/predict", json=dos_event)
    assert resp.status_code == 200
    data = resp.json()
    assert "attack_type" in data


def test_metrics_endpoint(trained_client: TestClient, sample_event: dict) -> None:
    trained_client.post("/predict", json=sample_event)
    resp = trained_client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_health" in data


def test_drift_endpoint(trained_client: TestClient) -> None:
    resp = trained_client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data


def test_feature_importance(trained_client: TestClient) -> None:
    resp = trained_client.get("/feature-importance")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_importance" in data


@pytest.mark.parametrize("protocol", ["tcp", "udp", "icmp", "other"])
def test_predict_various_protocols(trained_client: TestClient, protocol: str) -> None:
    event = {"protocol": protocol, "payload_bytes": 100, "duration_ms": 50.0}
    resp = trained_client.post("/predict", json=event)
    assert resp.status_code == 200


@pytest.mark.parametrize("payload_bytes", [0, 1, 100, 65535])
def test_predict_various_payload_sizes(trained_client: TestClient, payload_bytes: int) -> None:
    event = {"protocol": "tcp", "payload_bytes": payload_bytes}
    resp = trained_client.post("/predict", json=event)
    assert resp.status_code == 200


def test_predict_invalid_negative_bytes(client: TestClient) -> None:
    event = {"protocol": "tcp", "payload_bytes": -1}
    resp = client.post("/predict", json=event)
    assert resp.status_code == 422


def test_predict_invalid_serror_rate(client: TestClient) -> None:
    event = {"protocol": "tcp", "serror_rate": 1.5}
    resp = client.post("/predict", json=event)
    assert resp.status_code == 422


def test_predict_response_has_similar_patterns(trained_client: TestClient, sample_event: dict) -> None:
    resp = trained_client.post("/predict", json=sample_event)
    assert resp.status_code == 200
    data = resp.json()
    assert "similar_patterns" in data
    assert isinstance(data["similar_patterns"], list)


def test_health_response_has_version(client: TestClient) -> None:
    resp = client.get("/health")
    data = resp.json()
    assert "version" in data


def test_correlation_id_header(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Correlation-ID": "test-123"})
    assert resp.headers.get("X-Correlation-ID") == "test-123"


def test_predict_untrained_returns_503(client, sample_event, monkeypatch, tmp_path) -> None:
    import app.model as model_module
    from app.cache import clear_cache

    clear_cache()
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(model_module, "LABEL_ENCODER_PATH", tmp_path / "missing_le.pkl")
    resp = client.post("/predict", json=sample_event)
    assert resp.status_code == 503
