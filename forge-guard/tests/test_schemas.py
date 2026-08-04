"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BatchPredictionResponse, BatchSensorInput


def test_batch_input_accepts_valid_readings():
    payload = BatchSensorInput(readings=[{"temperature": 75.0, "pressure": 50.0}])
    assert len(payload.readings) == 1


def test_batch_input_rejects_empty_list():
    with pytest.raises(ValidationError):
        BatchSensorInput(readings=[])


def test_batch_input_rejects_over_100():
    with pytest.raises(ValidationError):
        BatchSensorInput(readings=[{"temperature": 75.0}] * 101)


def test_batch_response_round_trip():
    resp = BatchPredictionResponse(
        predictions=[{"prediction": 0, "defect_probability": 0.1}],
        count=1,
        model_version="1.0.0",
    )
    data = resp.model_dump()
    assert data["count"] == 1
    assert data["predictions"][0]["prediction"] == 0


@pytest.mark.parametrize("n", [1, 50, 100])
def test_batch_input_boundary_sizes(n: int):
    payload = BatchSensorInput(readings=[{"temperature": 75.0}] * n)
    assert len(payload.readings) == n


def test_batch_response_multiple_predictions():
    preds = [{"prediction": i % 2, "defect_probability": 0.1 * i} for i in range(5)]
    resp = BatchPredictionResponse(predictions=preds, count=5, model_version="1.0.0")
    assert resp.count == 5
    assert len(resp.predictions) == 5


def test_batch_input_preserves_sensor_values():
    reading = {"temperature": 82.5, "pressure": 53.1, "vibration": 3.7}
    payload = BatchSensorInput(readings=[reading])
    assert payload.readings[0]["temperature"] == 82.5


def test_batch_response_model_version_propagated():
    resp = BatchPredictionResponse(predictions=[], count=0, model_version="2.3.1")
    assert resp.model_version == "2.3.1"


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0", "1.1.0-beta"])
def test_batch_response_accepts_various_version_strings(version):
    resp = BatchPredictionResponse(predictions=[], count=0, model_version=version)
    assert resp.model_version == version
