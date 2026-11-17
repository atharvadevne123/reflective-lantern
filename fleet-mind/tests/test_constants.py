"""Tests for Fleet-Mind domain constants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.constants import (
    BATTERY_VOLTAGE_CRITICAL_V,
    BATTERY_VOLTAGE_LOW_V,
    BATTERY_VOLTAGE_NOMINAL_V,
    BRAKE_PAD_CRITICAL_MM,
    BRAKE_PAD_NEW_MM,
    BRAKE_PAD_WORN_THRESHOLD_MM,
    DEFAULT_RATE_LIMIT_RPM,
    DRIFT_BASELINE_SAMPLES,
    DRIFT_MIN_OBSERVATIONS,
    ENGINE_TEMP_CRITICAL_C,
    ENGINE_TEMP_NORMAL_MAX_C,
    ENGINE_TEMP_WARNING_C,
    OIL_PRESSURE_CRITICAL_BAR,
    OIL_PRESSURE_MIN_BAR,
    RUL_KM_PER_MM_BRAKE,
    SCALER_WARMUP_ROWS,
    TIRE_PRESSURE_HIGH_BAR,
    TIRE_PRESSURE_LOW_BAR,
    TIRE_PRESSURE_NOMINAL_BAR,
)


def test_brake_pad_ordering():
    assert BRAKE_PAD_CRITICAL_MM < BRAKE_PAD_WORN_THRESHOLD_MM < BRAKE_PAD_NEW_MM


def test_engine_temp_ordering():
    assert ENGINE_TEMP_NORMAL_MAX_C < ENGINE_TEMP_WARNING_C < ENGINE_TEMP_CRITICAL_C


def test_battery_voltage_ordering():
    assert BATTERY_VOLTAGE_CRITICAL_V < BATTERY_VOLTAGE_LOW_V < BATTERY_VOLTAGE_NOMINAL_V


def test_oil_pressure_critical_less_than_min():
    assert OIL_PRESSURE_CRITICAL_BAR < OIL_PRESSURE_MIN_BAR


def test_tire_pressure_ordering():
    assert TIRE_PRESSURE_LOW_BAR < TIRE_PRESSURE_NOMINAL_BAR < TIRE_PRESSURE_HIGH_BAR


def test_rul_positive():
    assert RUL_KM_PER_MM_BRAKE > 0


def test_drift_baseline_greater_than_min_observations():
    assert DRIFT_BASELINE_SAMPLES > DRIFT_MIN_OBSERVATIONS


def test_default_rate_limit_positive():
    assert DEFAULT_RATE_LIMIT_RPM > 0


def test_scaler_warmup_rows_positive():
    assert SCALER_WARMUP_ROWS > 0


@pytest.mark.parametrize("constant", [
    BRAKE_PAD_NEW_MM,
    BRAKE_PAD_WORN_THRESHOLD_MM,
    BRAKE_PAD_CRITICAL_MM,
    ENGINE_TEMP_NORMAL_MAX_C,
    ENGINE_TEMP_WARNING_C,
    ENGINE_TEMP_CRITICAL_C,
    OIL_PRESSURE_MIN_BAR,
    OIL_PRESSURE_CRITICAL_BAR,
    BATTERY_VOLTAGE_NOMINAL_V,
    BATTERY_VOLTAGE_LOW_V,
    BATTERY_VOLTAGE_CRITICAL_V,
    RUL_KM_PER_MM_BRAKE,
])
def test_constants_are_positive(constant):
    assert constant > 0
