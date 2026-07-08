"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import math

import pytest

from app.features import (
    FEATURE_COLUMNS,
    PEAK_HOURS,
    ROAD_TYPE_MAP,
    build_feature_dataframe,
    build_feature_vector,
    compute_lag_features,
    compute_rolling_features,
    encode_road_type,
    is_peak_hour,
)


def _minimal_raw(hour: int = 8, dow: int = 1) -> dict:
    return {"hour": hour, "day_of_week": dow, "vehicle_count": 1000, "avg_speed_kmh": 50.0}


def test_build_feature_vector_all_columns() -> None:
    fv = build_feature_vector(_minimal_raw())
    for col in FEATURE_COLUMNS:
        assert col in fv, f"Missing: {col}"


def test_is_peak_hour_morning_true() -> None:
    assert is_peak_hour(8) == 1
    assert is_peak_hour(9) == 1


def test_is_peak_hour_evening_true() -> None:
    assert is_peak_hour(17) == 1
    assert is_peak_hour(19) == 1


def test_is_peak_hour_off_peak_false() -> None:
    assert is_peak_hour(3) == 0
    assert is_peak_hour(13) == 0


@pytest.mark.parametrize("road_type,code", list(ROAD_TYPE_MAP.items()))
def test_encode_road_type_known(road_type: str, code: int) -> None:
    assert encode_road_type(road_type) == code


def test_encode_road_type_unknown_defaults_to_local() -> None:
    assert encode_road_type("gravel") == 3


@pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
def test_cyclical_hour_features_in_range(hour: int) -> None:
    fv = build_feature_vector(_minimal_raw(hour=hour))
    assert -1.0 <= fv["hour_sin"] <= 1.0
    assert -1.0 <= fv["hour_cos"] <= 1.0


def test_weekend_flag_saturday() -> None:
    fv = build_feature_vector(_minimal_raw(dow=6))
    assert fv["is_weekend"] == 1


def test_weekend_flag_monday() -> None:
    fv = build_feature_vector(_minimal_raw(dow=0))
    assert fv["is_weekend"] == 0


def test_lag_features_default_fallback() -> None:
    lag = compute_lag_features(1000.0, 50.0)
    assert lag["lag_1h"] == pytest.approx(950.0)
    assert lag["lag_delta_1h"] == pytest.approx(50.0)


def test_lag_ratio_positive() -> None:
    lag = compute_lag_features(1000.0, 50.0)
    assert lag["lag_ratio_1h"] > 0


def test_rolling_volume_vs_daily_avg() -> None:
    rolling = compute_rolling_features(1200.0, rolling_mean_24h=1000.0)
    assert rolling["volume_vs_daily_avg"] == pytest.approx(1.2, rel=0.01)


def test_build_feature_dataframe_batch_length() -> None:
    records = [_minimal_raw(hour=h) for h in range(5)]
    df = build_feature_dataframe(records)
    assert len(df) == 5


def test_build_feature_dataframe_has_all_columns() -> None:
    df = build_feature_dataframe([_minimal_raw()])
    assert set(FEATURE_COLUMNS).issubset(set(df.columns))


def test_speed_volume_ratio_numerics() -> None:
    fv = build_feature_vector(
        {"hour": 10, "day_of_week": 2, "vehicle_count": 500, "avg_speed_kmh": 50.0}
    )
    assert math.isfinite(fv["speed_volume_ratio"])
    assert fv["speed_volume_ratio"] > 0


def test_peak_hours_constant_membership() -> None:
    assert 7 in PEAK_HOURS
    assert 19 in PEAK_HOURS
    assert 12 not in PEAK_HOURS


@pytest.mark.parametrize("vehicle_count", [0, 1, 9999])
def test_feature_vector_extreme_vehicle_counts(vehicle_count: int) -> None:
    fv = build_feature_vector(
        {"hour": 12, "day_of_week": 3, "vehicle_count": vehicle_count, "avg_speed_kmh": 40.0}
    )
    assert all(math.isfinite(v) for v in fv.values())


def test_feature_vector_zero_speed_finite() -> None:
    fv = build_feature_vector(
        {"hour": 12, "day_of_week": 3, "vehicle_count": 100, "avg_speed_kmh": 0.0}
    )
    assert math.isfinite(fv["speed_volume_ratio"])
