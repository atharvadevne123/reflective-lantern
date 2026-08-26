"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["version"] == "1.0.0"


def test_predict_normal_flow(client: TestClient, sample_normal_flow: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_normal_flow)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert isinstance(data["is_attack"], bool)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_dos_flow(client: TestClient, sample_dos_flow: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_dos_flow)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_class"] in ["normal", "dos", "probe", "r2l", "u2r"]


def test_predict_returns_correlation_id(client: TestClient, sample_normal_flow: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_normal_flow)
    assert resp.status_code == 200
    data = resp.json()
    assert "correlation_id" in data
    assert len(data["correlation_id"]) > 0


def test_predict_class_probabilities(client: TestClient, sample_normal_flow: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_normal_flow)
    assert resp.status_code == 200
    probs = resp.json()["class_probabilities"]
    assert set(probs.keys()) == {"normal", "dos", "probe", "r2l", "u2r"}
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_predict_invalid_protocol(client: TestClient, sample_normal_flow: dict) -> None:
    bad_flow = {**sample_normal_flow, "protocol_type": "banana"}
    resp = client.post("/api/v1/predict", json=bad_flow)
    assert resp.status_code == 422


def test_predict_negative_bytes_rejected(client: TestClient, sample_normal_flow: dict) -> None:
    bad_flow = {**sample_normal_flow, "src_bytes": -100}
    resp = client.post("/api/v1/predict", json=bad_flow)
    assert resp.status_code == 422


def test_metrics_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_count" in data
    assert isinstance(data["prediction_count"], int)


def test_drift_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_threats_search(client: TestClient) -> None:
    resp = client.get("/api/v1/threats", params={"q": "dos flooding", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert isinstance(data["results"], list)


@pytest.mark.parametrize("protocol", ["tcp", "udp", "icmp"])
def test_predict_all_protocols(client: TestClient, sample_normal_flow: dict, protocol: str) -> None:
    flow = {**sample_normal_flow, "protocol_type": protocol}
    resp = client.post("/api/v1/predict", json=flow)
    assert resp.status_code == 200
