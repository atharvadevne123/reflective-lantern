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


def test_drift_check_result_schema():
    from app.schemas import DriftCheckResult

    r = DriftCheckResult(
        feature="temperature", ks_statistic=0.42, p_value=0.001, drift_detected=True
    )
    assert r.feature == "temperature"
    assert r.drift_detected is True


def test_model_summary_response_defaults():
    from app.schemas import ModelSummaryResponse

    r = ModelSummaryResponse(model_version="1.0.0", total=0)
    assert r.defects == 0
    assert r.defect_rate == 0.0
    assert r.avg_defect_probability == 0.0


def test_model_summary_response_with_data():
    from app.schemas import ModelSummaryResponse

    r = ModelSummaryResponse(
        model_version="1.0.0",
        total=100,
        defects=10,
        defect_rate=0.1,
        avg_defect_probability=0.12,
    )
    assert r.total == 100
    assert r.defect_rate == 0.1


def test_retraining_trigger_response():
    from app.schemas import RetrainingTriggerResponse

    r = RetrainingTriggerResponse(status="accepted", message="Pipeline started")
    assert r.status == "accepted"
    assert "Pipeline" in r.message


def test_drift_check_result_no_drift():
    from app.schemas import DriftCheckResult

    r = DriftCheckResult(feature="humidity", ks_statistic=0.05, p_value=0.75, drift_detected=False)
    assert r.drift_detected is False
    assert r.p_value == 0.75
