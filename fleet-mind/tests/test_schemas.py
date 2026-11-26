"""Tests for Fleet-Mind Pydantic schemas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    BatchPredictionResponse,
    BatchTelemetryInput,
    PredictionResponse,
    TelemetryInput,
)


def _valid_reading(**overrides):
    base = {
        "vehicle_id": "VH-001",
        "speed_kmh": 80.0,
        "engine_temp_c": 90.0,
        "oil_pressure_bar": 3.0,
        "fuel_consumption_lph": 7.0,
        "battery_voltage_v": 12.5,
        "brake_pad_mm": 6.0,
        "tire_pressure_bar": 2.3,
        "rpm": 2500.0,
        "coolant_temp_c": 85.0,
        "transmission_temp_c": 70.0,
    }
    base.update(overrides)
    return base


def test_valid_telemetry_input_accepted():
    t = TelemetryInput(**_valid_reading())
    assert t.vehicle_id == "VH-001"


def test_vehicle_id_path_traversal_rejected():
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(vehicle_id="../etc/passwd"))


def test_vehicle_id_null_byte_rejected():
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(vehicle_id="VH\x00bad"))


def test_vehicle_id_empty_rejected():
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(vehicle_id=""))


def test_speed_negative_rejected():
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(speed_kmh=-1.0))


def test_speed_above_max_rejected():
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(speed_kmh=301.0))


def test_engine_temp_range_accepted():
    t = TelemetryInput(**_valid_reading(engine_temp_c=-40.0))
    assert t.engine_temp_c == -40.0


def test_prediction_response_fields():
    pr = PredictionResponse(
        vehicle_id="VH-001",
        maintenance_needed=True,
        anomaly_score=0.75,
        anomaly_type="brake",
        remaining_useful_life_km=5000.0,
        confidence=0.9,
        model_version="1.0.0",
    )
    assert pr.vehicle_id == "VH-001"
    assert pr.maintenance_needed is True


def test_batch_input_requires_at_least_one_reading():
    with pytest.raises(Exception):
        BatchTelemetryInput(readings=[])


def test_batch_input_accepts_single_reading():
    batch = BatchTelemetryInput(readings=[TelemetryInput(**_valid_reading())])
    assert len(batch.readings) == 1


def test_batch_prediction_response_counts():
    pr = PredictionResponse(
        vehicle_id="VH-001",
        maintenance_needed=False,
        anomaly_score=0.1,
        anomaly_type=None,
        remaining_useful_life_km=None,
        confidence=0.8,
        model_version="1.0.0",
    )
    batch_resp = BatchPredictionResponse(results=[pr], total=1, maintenance_flagged=0)
    assert batch_resp.total == 1


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("oil_pressure_bar", -1.0),
        ("battery_voltage_v", 25.0),
        ("brake_pad_mm", 25.0),
        ("tire_pressure_bar", 6.0),
    ],
)
def test_out_of_range_fields_rejected(field, bad_value):
    with pytest.raises(Exception):
        TelemetryInput(**_valid_reading(**{field: bad_value}))
