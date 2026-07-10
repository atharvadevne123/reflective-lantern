"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import AnomalyRequest, DriftRequest, EnergyReadingIn


def test_energy_reading_valid():
    r = EnergyReadingIn(
        building_id="bldg-001",
        timestamp="2025-06-01T14:00:00",
        hour=14,
        day_of_week=1,
        month=6,
        temperature_c=28.5,
        humidity_pct=60.0,
        occupancy=50,
        hvac_state=1,
        consumption_kwh=12.0,
    )
    assert r.building_id == "bldg-001"
    assert r.hour == 14


def test_energy_reading_invalid_hour():
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="b",
            timestamp="2025-01-01T00:00:00",
            hour=25,
            day_of_week=0,
            month=1,
            temperature_c=20.0,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


def test_energy_reading_invalid_building_id():
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="bad id!@#",
            timestamp="2025-01-01T00:00:00",
            hour=12,
            day_of_week=0,
            month=1,
            temperature_c=20.0,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


@pytest.mark.parametrize("temp", [-41.0, 61.0])
def test_energy_reading_invalid_temperature(temp):
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="b001",
            timestamp="2025-01-01T00:00:00",
            hour=12,
            day_of_week=0,
            month=1,
            temperature_c=temp,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


def test_drift_request_min_length():
    with pytest.raises(ValidationError):
        DriftRequest(current_values=[1.0, 2.0])


def test_anomaly_request_valid():
    r = AnomalyRequest(
        building_id="b-99",
        timestamp="2025-03-15T08:00:00",
        consumption_kwh=20.0,
        hour=8,
        day_of_week=0,
        month=3,
    )
    assert r.temperature_c == 20.0
    assert r.hvac_state == 0
