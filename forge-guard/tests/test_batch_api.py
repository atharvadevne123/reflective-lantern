"""Tests for batch prediction endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_batch_predict_single(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/api/v1/predict/batch", json={"readings": [sample_sensor_payload]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["prediction"] in (0, 1)


def test_batch_predict_multiple(client: TestClient, sample_sensor_payload: dict) -> None:
    readings = [sample_sensor_payload] * 5
    resp = client.post("/api/v1/predict/batch", json={"readings": readings})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5


def test_batch_predict_empty_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/predict/batch", json={"readings": []})
    assert resp.status_code == 422


def test_batch_predict_over_limit_rejected(client: TestClient, sample_sensor_payload: dict) -> None:
    readings = [sample_sensor_payload] * 101
    resp = client.post("/api/v1/predict/batch", json={"readings": readings})
    assert resp.status_code == 422


@pytest.mark.parametrize("n", [1, 3, 10])
def test_batch_predict_count_matches(
    client: TestClient, sample_sensor_payload: dict, n: int
) -> None:
    readings = [sample_sensor_payload] * n
    resp = client.post("/api/v1/predict/batch", json={"readings": readings})
    assert resp.status_code == 200
    assert resp.json()["count"] == n
