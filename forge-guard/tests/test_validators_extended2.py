"""More extended tests for forge-guard app/validators.py helper functions."""

from __future__ import annotations

import pytest

from app.validators import (
    clamp_reading,
    field_statistics,
    missing_rate,
    sensor_z_score,
    zscore_outlier,
)


class TestSensorZScore:
    def test_zero_when_at_mean(self) -> None:
        assert sensor_z_score(5.0, mean=5.0, std=1.0) == pytest.approx(0.0)

    def test_one_when_one_std_above(self) -> None:
        assert sensor_z_score(6.0, mean=5.0, std=1.0) == pytest.approx(1.0)

    def test_negative_below_mean(self) -> None:
        assert sensor_z_score(4.0, mean=5.0, std=1.0) == pytest.approx(-1.0)

    def test_zero_std_no_error(self) -> None:
        try:
            result = sensor_z_score(5.0, mean=5.0, std=0.0)
            assert result == 0.0 or result != result  # 0 or NaN
        except (ZeroDivisionError, ValueError):
            pass


class TestClampReading:
    def test_value_in_range_unchanged(self) -> None:
        assert clamp_reading(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_below_lo_clamps(self) -> None:
        assert clamp_reading(-1.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_above_hi_clamps(self) -> None:
        assert clamp_reading(15.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_at_boundary(self) -> None:
        assert clamp_reading(0.0, 0.0, 10.0) == pytest.approx(0.0)
        assert clamp_reading(10.0, 0.0, 10.0) == pytest.approx(10.0)


class TestZscoreOutlier:
    def test_no_outliers_in_uniform_data(self) -> None:
        values = [1.0] * 20
        flags = zscore_outlier(values)
        assert all(f is False for f in flags)

    def test_detects_spike(self) -> None:
        values = [1.0] * 19 + [100.0]
        flags = zscore_outlier(values)
        assert flags[-1] is True

    def test_length_matches_input(self) -> None:
        values = [1.0, 2.0, 3.0, 100.0]
        flags = zscore_outlier(values)
        assert len(flags) == len(values)

    def test_custom_threshold(self) -> None:
        values = [0.0, 0.0, 0.0, 10.0]
        flags_strict = zscore_outlier(values, threshold=1.0)
        assert any(flags_strict)


class TestMissingRate:
    def test_zero_when_all_present(self) -> None:
        records = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        assert missing_rate(records, "x") == pytest.approx(0.0)

    def test_one_when_all_missing(self) -> None:
        records = [{}, {}, {}]
        assert missing_rate(records, "x") == pytest.approx(1.0)

    def test_half_missing(self) -> None:
        records = [{"x": 1.0}, {}, {"x": 3.0}, {}]
        assert missing_rate(records, "x") == pytest.approx(0.5)

    def test_empty_list_zero(self) -> None:
        assert missing_rate([], "x") == pytest.approx(0.0)


class TestFieldStatistics:
    def test_returns_dict(self) -> None:
        records = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        stats = field_statistics(records, "x")
        assert isinstance(stats, dict)

    def test_contains_mean(self) -> None:
        records = [{"x": 1.0}, {"x": 3.0}]
        stats = field_statistics(records, "x")
        assert "mean" in stats
        assert stats["mean"] == pytest.approx(2.0)

    def test_contains_min_max(self) -> None:
        records = [{"x": 1.0}, {"x": 5.0}, {"x": 3.0}]
        stats = field_statistics(records, "x")
        assert stats.get("min") == pytest.approx(1.0)
        assert stats.get("max") == pytest.approx(5.0)
