"""Tests for app/constants.py."""

from __future__ import annotations

import pytest

from app.constants import (
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DRIFT_P_THRESHOLD,
    EFFICIENCY_GRADE_A_THRESHOLD,
    EFFICIENCY_GRADE_B_THRESHOLD,
    EFFICIENCY_GRADE_C_THRESHOLD,
    GRID_INTENSITY_KG_PER_KWH,
    IQR_MULTIPLIER,
    MAX_CONSUMPTION_KWH,
    MAX_FORECAST_HORIZON,
    MAX_HUMIDITY_PCT,
    MAX_TEMPERATURE_C,
    MIN_CONSUMPTION_KWH,
    MIN_HUMIDITY_PCT,
    MIN_REFERENCE_SIZE,
    MIN_TEMPERATURE_C,
    REFERENCE_WINDOW_SIZE,
    ZSCORE_THRESHOLD,
)


class TestAnomalyConstants:
    def test_zscore_threshold_positive(self) -> None:
        assert ZSCORE_THRESHOLD > 0.0

    def test_iqr_multiplier_positive(self) -> None:
        assert IQR_MULTIPLIER > 0.0

    def test_min_reference_size_at_least_two(self) -> None:
        assert MIN_REFERENCE_SIZE >= 2


class TestDriftConstants:
    def test_p_threshold_in_unit_interval(self) -> None:
        assert 0.0 < DRIFT_P_THRESHOLD < 1.0

    def test_reference_window_size_positive(self) -> None:
        assert REFERENCE_WINDOW_SIZE > 0


class TestEnergyConstants:
    def test_max_consumption_gt_min(self) -> None:
        assert MAX_CONSUMPTION_KWH > MIN_CONSUMPTION_KWH

    def test_temperature_range_valid(self) -> None:
        assert MIN_TEMPERATURE_C < MAX_TEMPERATURE_C

    def test_humidity_range_0_to_100(self) -> None:
        assert MIN_HUMIDITY_PCT == pytest.approx(0.0)
        assert MAX_HUMIDITY_PCT == pytest.approx(100.0)

    def test_forecast_horizon_one_week(self) -> None:
        assert MAX_FORECAST_HORIZON == 168


class TestGridIntensity:
    def test_all_values_positive(self) -> None:
        for region, intensity in GRID_INTENSITY_KG_PER_KWH.items():
            assert intensity > 0.0, f"Region {region} has non-positive intensity"

    @pytest.mark.parametrize("region", ["northeast", "midwest", "south", "west", "default"])
    def test_known_regions_present(self, region: str) -> None:
        assert region in GRID_INTENSITY_KG_PER_KWH

    def test_default_region_present(self) -> None:
        assert "default" in GRID_INTENSITY_KG_PER_KWH


class TestCacheConstants:
    def test_ttl_positive(self) -> None:
        assert DEFAULT_CACHE_TTL_SECONDS > 0

    def test_max_size_positive(self) -> None:
        assert DEFAULT_CACHE_MAX_SIZE > 0


class TestEfficiencyGradeThresholds:
    def test_a_gt_b_gt_c(self) -> None:
        assert EFFICIENCY_GRADE_A_THRESHOLD > EFFICIENCY_GRADE_B_THRESHOLD
        assert EFFICIENCY_GRADE_B_THRESHOLD > EFFICIENCY_GRADE_C_THRESHOLD

    def test_all_in_unit_interval(self) -> None:
        for val in [EFFICIENCY_GRADE_A_THRESHOLD, EFFICIENCY_GRADE_B_THRESHOLD, EFFICIENCY_GRADE_C_THRESHOLD]:
            assert 0.0 < val < 1.0


def test_rate_limit_positive() -> None:
    assert DEFAULT_RATE_LIMIT_PER_MINUTE > 0
