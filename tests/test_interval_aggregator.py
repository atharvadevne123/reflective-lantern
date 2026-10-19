"""Tests for app.interval_aggregator module."""

from __future__ import annotations

import pytest

from app.interval_aggregator import (
    AggregatedInterval,
    aggregate,
    detect_outliers_iqr,
    percentile,
    resample_mean,
    resample_sum,
    rolling_average,
)


class TestAggregate:
    def test_returns_dataclass(self) -> None:
        result = aggregate([1.0, 2.0, 3.0])
        assert isinstance(result, AggregatedInterval)

    def test_count(self) -> None:
        assert aggregate([10.0, 20.0, 30.0]).count == 3

    def test_total(self) -> None:
        assert aggregate([10.0, 20.0, 30.0]).total == pytest.approx(60.0)

    def test_mean(self) -> None:
        assert aggregate([10.0, 20.0, 30.0]).mean == pytest.approx(20.0)

    def test_min_max(self) -> None:
        r = aggregate([5.0, 15.0, 10.0])
        assert r.minimum == 5.0
        assert r.maximum == 15.0

    def test_median(self) -> None:
        assert aggregate([1.0, 2.0, 3.0]).median == pytest.approx(2.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate([])

    @pytest.mark.parametrize(
        "values,expected_mean",
        [
            ([2.0, 2.0, 2.0], 2.0),
            ([0.0, 10.0], 5.0),
            ([1.0, 3.0, 5.0, 7.0], 4.0),
        ],
    )
    def test_mean_parametrized(self, values: list[float], expected_mean: float) -> None:
        assert aggregate(values).mean == pytest.approx(expected_mean)


class TestResampleSum:
    def test_bucket_size_one_unchanged(self) -> None:
        assert resample_sum([1.0, 2.0, 3.0], 1) == pytest.approx([1.0, 2.0, 3.0])

    def test_bucket_size_equals_length(self) -> None:
        assert resample_sum([1.0, 2.0, 3.0], 3) == pytest.approx([6.0])

    def test_96_to_24_readings(self) -> None:
        vals = [1.0] * 96
        resampled = resample_sum(vals, 4)
        assert len(resampled) == 24
        assert all(v == pytest.approx(4.0) for v in resampled)

    def test_remainder_bucket_included(self) -> None:
        result = resample_sum([1.0, 2.0, 3.0, 4.0, 5.0], 2)
        assert len(result) == 3
        assert result[-1] == pytest.approx(5.0)

    def test_zero_bucket_size_raises(self) -> None:
        with pytest.raises(ValueError):
            resample_sum([1.0, 2.0], 0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            resample_sum([], 2)


class TestResampleMean:
    def test_flat_series_unchanged(self) -> None:
        result = resample_mean([5.0, 5.0, 5.0, 5.0], 2)
        assert all(v == pytest.approx(5.0) for v in result)

    def test_bucket_size_one(self) -> None:
        vals = [1.0, 2.0, 3.0]
        assert resample_mean(vals, 1) == pytest.approx(vals)

    def test_zero_bucket_raises(self) -> None:
        with pytest.raises(ValueError):
            resample_mean([1.0], 0)

    @pytest.mark.parametrize(
        "values,bucket,expected",
        [
            ([2.0, 4.0], 2, [3.0]),
            ([1.0, 3.0, 5.0, 7.0], 2, [2.0, 6.0]),
        ],
    )
    def test_parametrized(
        self, values: list[float], bucket: int, expected: list[float]
    ) -> None:
        assert resample_mean(values, bucket) == pytest.approx(expected)


class TestRollingAverage:
    def test_window_one_unchanged(self) -> None:
        vals = [1.0, 2.0, 3.0]
        assert rolling_average(vals, 1) == pytest.approx(vals)

    def test_window_equals_length(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0]
        result = rolling_average(vals, len(vals))
        assert result[-1] == pytest.approx(2.5)

    def test_expanding_start(self) -> None:
        result = rolling_average([2.0, 4.0, 6.0], 3)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(4.0)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_average([1.0], 0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_average([], 2)


class TestPercentile:
    def test_p0_is_minimum(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 0) == pytest.approx(1.0)

    def test_p100_is_maximum(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 100) == pytest.approx(3.0)

    def test_p50_median(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], 101)

    @pytest.mark.parametrize("p", [25.0, 50.0, 75.0])
    def test_quartiles_non_negative(self, p: float) -> None:
        assert percentile([0.0, 1.0, 2.0, 3.0, 4.0], p) >= 0


class TestDetectOutliersIQR:
    def test_no_outliers(self) -> None:
        assert detect_outliers_iqr([1.0, 2.0, 3.0, 4.0, 5.0]) == []

    def test_high_outlier_detected(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 1000.0]
        outliers = detect_outliers_iqr(values)
        assert 4 in outliers

    def test_too_few_values_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_outliers_iqr([1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 10.0, 10.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
    )
    def test_returns_list(self, values: list[float]) -> None:
        result = detect_outliers_iqr(values)
        assert isinstance(result, list)
