"""Extended tests for Fleet-Mind Pydantic schemas and validators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    BatchPredictionResponse,
    BatchTelemetryInput,
    PredictionResponse,
    TelemetryInput,
)


def _valid_reading(**overrides) -> dict:
    base = dict(
        vehicle_id="VH-001",
        speed_kmh=60.0,
        engine_temp_c=90.0,
        oil_pressure_bar=3.5,
        fuel_consumption_lph=8.0,
        battery_voltage_v=12.5,
        brake_pad_mm=6.0,
        tire_pressure_bar=2.3,
        rpm=2800.0,
        coolant_temp_c=85.0,
        transmission_temp_c=70.0,
    )
    base.update(overrides)
    return base


def test_valid_telemetry_input():
    t = TelemetryInput(**_valid_reading())
    assert t.vehicle_id == "VH-001"


def test_speed_at_zero_valid():
    t = TelemetryInput(**_valid_reading(speed_kmh=0.0))
    assert t.speed_kmh == 0.0


def test_speed_above_max_raises():
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(speed_kmh=301.0))


def test_engine_temp_min_valid():
    t = TelemetryInput(**_valid_reading(engine_temp_c=-40.0))
    assert t.engine_temp_c == -40.0


def test_engine_temp_above_max_raises():
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(engine_temp_c=201.0))


def test_vehicle_id_stripped_of_whitespace():
    t = TelemetryInput(**_valid_reading(vehicle_id="  VH-002  "))
    assert t.vehicle_id == "VH-002"


def test_vehicle_id_null_byte_raises():
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(vehicle_id="VH\x00001"))


def test_vehicle_id_path_traversal_raises():
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(vehicle_id="../etc/passwd"))


def test_rpm_at_zero_valid():
    t = TelemetryInput(**_valid_reading(rpm=0.0))
    assert t.rpm == 0.0


def test_rpm_above_max_raises():
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(rpm=10001.0))


def test_batch_telemetry_requires_at_least_one():
    with pytest.raises(ValidationError):
        BatchTelemetryInput(readings=[])


def test_prediction_response_valid():
    pr = PredictionResponse(
        vehicle_id="VH-001",
        maintenance_needed=False,
        anomaly_score=0.1,
        anomaly_type=None,
        remaining_useful_life_km=50000.0,
        confidence=0.95,
        model_version="1.0",
    )
    assert pr.vehicle_id == "VH-001"


def test_batch_prediction_response_total():
    pr = PredictionResponse(
        vehicle_id="v1",
        maintenance_needed=True,
        anomaly_score=0.9,
        anomaly_type="wear",
        remaining_useful_life_km=None,
        confidence=0.8,
        model_version="1.0",
    )
    bpr = BatchPredictionResponse(results=[pr], total=1, maintenance_flagged=1)
    assert bpr.total == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("oil_pressure_bar", -0.1),
        ("battery_voltage_v", -1.0),
        ("brake_pad_mm", -0.1),
        ("tire_pressure_bar", -0.1),
    ],
)
def test_negative_physical_values_raise(field, value):
    with pytest.raises(ValidationError):
        TelemetryInput(**_valid_reading(**{field: value}))
