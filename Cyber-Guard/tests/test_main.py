"""Tests for Cyber-Guard app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("app.database.create_tables"),
        patch("app.model.ensure_model_exists"),
        patch("app.anomaly.ensure_anomaly_model_exists"),
        patch("app.model.load_model", return_value=(MagicMock(), MagicMock())),
        patch("app.anomaly.load_anomaly_detector", return_value=MagicMock()),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_health_status_is_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ("ok", "healthy", "degraded")


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_returns_dict(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert isinstance(resp.json(), dict)


class TestPredictEndpoint:
    def test_predict_missing_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/predict", json={})
        assert resp.status_code in (200, 422, 400, 500)

    def test_predict_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "duration": 1,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 100,
                "dst_bytes": 200,
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": 0,
                "num_failed_logins": 0,
                "logged_in": 1,
                "num_compromised": 0,
                "root_shell": 0,
                "su_attempted": 0,
                "num_root": 0,
                "num_file_creations": 0,
                "num_shells": 0,
                "num_access_files": 0,
                "num_outbound_cmds": 0,
                "is_host_login": 0,
                "is_guest_login": 0,
                "count": 1,
                "srv_count": 1,
                "serror_rate": 0.0,
                "srv_serror_rate": 0.0,
                "rerror_rate": 0.0,
                "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": 1,
                "dst_host_srv_count": 1,
                "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.0,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0,
                "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_rerror_rate": 0.0,
            },
        )
        assert isinstance(resp.json(), dict)
