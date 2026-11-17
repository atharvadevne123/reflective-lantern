"""Tests for Fleet-Mind custom exception types."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.exceptions import (
    DriftCheckError,
    FleetMindError,
    ModelNotReadyError,
    ModelTrainingError,
    PredictionError,
)


def test_fleet_mind_error_is_exception():
    err = FleetMindError("base error")
    assert isinstance(err, Exception)
    assert str(err) == "base error"


def test_model_not_ready_default_message():
    err = ModelNotReadyError()
    assert "not ready" in str(err).lower()


def test_model_not_ready_custom_message():
    err = ModelNotReadyError("custom message")
    assert str(err) == "custom message"


def test_model_not_ready_is_fleet_mind_error():
    assert isinstance(ModelNotReadyError(), FleetMindError)


def test_prediction_error_message_contains_vehicle_id():
    cause = ValueError("bad input")
    err = PredictionError("VH-001", cause)
    assert "VH-001" in str(err)


def test_prediction_error_stores_attributes():
    cause = RuntimeError("fail")
    err = PredictionError("VH-XYZ", cause)
    assert err.vehicle_id == "VH-XYZ"
    assert err.cause is cause


def test_prediction_error_is_fleet_mind_error():
    assert isinstance(PredictionError("v", Exception()), FleetMindError)


def test_drift_check_error_is_fleet_mind_error():
    err = DriftCheckError("drift failed")
    assert isinstance(err, FleetMindError)


def test_model_training_error_message():
    err = ModelTrainingError("training failed")
    assert "training failed" in str(err)


def test_model_training_error_with_cause():
    cause = OSError("disk full")
    err = ModelTrainingError("training failed", cause=cause)
    assert err.cause is cause


def test_model_training_error_no_cause():
    err = ModelTrainingError("failed")
    assert err.cause is None


@pytest.mark.parametrize("exc_class", [FleetMindError, ModelNotReadyError, DriftCheckError])
def test_exceptions_can_be_raised_and_caught(exc_class):
    with pytest.raises(FleetMindError):
        raise exc_class("test")
