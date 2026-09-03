"""Tests for app/load_profile.py."""

from __future__ import annotations

import pytest

from app.load_profile import (
    FLAT_LOAD_FACTOR_THRESHOLD,
    PEAKY_LOAD_FACTOR_THRESHOLD,
    base_load,
    build_load_profile,
    classify_profile,
    demand_variability,
    load_factor,
    max_ramp_rate,
    night_load_fraction,
    peak_to_average_ratio,
)

FLAT = [5.0] * 24
PEAKY = [1.0] * 20 + [20.0, 22.0, 21.0, 1.0]
RAMP = [1.0, 2.0, 8.0, 3.0]


class TestBaseLoad:
    def test_constant_series(self) -> None:
        assert base_load(FLAT) == pytest.approx(5.0)

    def test_reads_low_tail_not_minimum(self) -> None:
        # With 24 entries and percentile 0.10, index 2 of the sorted series.
        assert base_load([0.0, 1.0, 2.0, 3.0, 4.0], percentile=0.4) == pytest.approx(2.0)

    def test_single_value_series(self) -> None:
        assert base_load([7.5]) == pytest.approx(7.5)

    def test_percentile_one_clamps_to_last_index(self) -> None:
        assert base_load([1.0, 2.0, 3.0], percentile=1.0) == pytest.approx(3.0)

    def test_base_load_never_exceeds_peak(self) -> None:
        assert base_load(PEAKY) <= max(PEAKY)

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            base_load([])

    @pytest.mark.parametrize("percentile", [-0.1, 1.5])
    def test_invalid_percentile_rejected(self, percentile: float) -> None:
        with pytest.raises(ValueError, match="percentile must be in 0-1"):
            base_load(FLAT, percentile=percentile)


class TestLoadFactor:
    def test_flat_series_is_one(self) -> None:
        assert load_factor(FLAT) == pytest.approx(1.0)

    def test_peaky_series_is_low(self) -> None:
        assert load_factor(PEAKY) < PEAKY_LOAD_FACTOR_THRESHOLD

    def test_all_zero_series_returns_zero(self) -> None:
        assert load_factor([0.0] * 10) == 0.0

    def test_bounded_between_zero_and_one(self) -> None:
        for series in (FLAT, PEAKY, RAMP):
            assert 0.0 <= load_factor(series) <= 1.0

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            load_factor([])


class TestPeakToAverageRatio:
    def test_flat_series_is_one(self) -> None:
        assert peak_to_average_ratio(FLAT) == pytest.approx(1.0)

    def test_peaky_series_exceeds_one(self) -> None:
        assert peak_to_average_ratio(PEAKY) > 1.0

    def test_all_zero_series_returns_zero(self) -> None:
        assert peak_to_average_ratio([0.0] * 10) == 0.0

    def test_is_reciprocal_of_load_factor(self) -> None:
        assert peak_to_average_ratio(RAMP) == pytest.approx(1.0 / load_factor(RAMP), rel=1e-3)

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            peak_to_average_ratio([])


class TestMaxRampRate:
    def test_largest_step_is_found(self) -> None:
        assert max_ramp_rate(RAMP) == pytest.approx(6.0)

    def test_flat_series_has_no_ramp(self) -> None:
        assert max_ramp_rate(FLAT) == 0.0

    def test_downward_ramp_counted_by_magnitude(self) -> None:
        assert max_ramp_rate([10.0, 1.0]) == pytest.approx(9.0)

    @pytest.mark.parametrize("series", [[], [4.0]])
    def test_short_series_returns_zero(self, series: list[float]) -> None:
        assert max_ramp_rate(series) == 0.0


class TestClassifyProfile:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, "flat"),
            (FLAT_LOAD_FACTOR_THRESHOLD, "flat"),
            (0.60, "moderate"),
            (PEAKY_LOAD_FACTOR_THRESHOLD, "moderate"),
            (0.20, "peaky"),
            (0.0, "peaky"),
        ],
    )
    def test_buckets(self, value: float, expected: str) -> None:
        assert classify_profile(value) == expected


class TestBuildLoadProfile:
    def test_flat_building_classified_flat(self) -> None:
        profile = build_load_profile(FLAT)
        assert profile.profile_class == "flat"
        assert profile.load_factor == pytest.approx(1.0)
        assert profile.max_ramp_kwh == 0.0

    def test_peaky_building_classified_peaky(self) -> None:
        profile = build_load_profile(PEAKY)
        assert profile.profile_class == "peaky"
        assert profile.peak_kwh == pytest.approx(22.0)

    def test_mean_lies_between_base_and_peak(self) -> None:
        profile = build_load_profile(PEAKY)
        assert profile.base_load_kwh <= profile.mean_kwh <= profile.peak_kwh

    def test_class_matches_standalone_classifier(self) -> None:
        profile = build_load_profile(RAMP)
        assert profile.profile_class == classify_profile(profile.load_factor)

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_load_profile([])


class TestLoadProfileExtensions:
    def test_demand_variability_flat_is_zero(self):
        assert demand_variability([5.0] * 24) == pytest.approx(0.0, abs=0.001)

    def test_demand_variability_volatile_series(self):
        series = [1.0, 10.0, 1.0, 10.0] * 6
        assert demand_variability(series) > 0.5

    def test_demand_variability_single_value(self):
        assert demand_variability([5.0]) == 0.0

    def test_night_load_fraction_all_night(self):
        hourly = [0.0] * 6 + [0.0] * 16 + [10.0] * 2
        frac = night_load_fraction(hourly, night_hours=(22, 6))
        assert frac == pytest.approx(1.0, abs=0.01)

    def test_night_load_fraction_short_series(self):
        assert night_load_fraction([1.0] * 10) == 0.0

    def test_night_load_fraction_zero_total(self):
        assert night_load_fraction([0.0] * 24) == 0.0
