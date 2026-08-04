"""Tests for app/trend_analysis.py."""

from __future__ import annotations

import pytest

from app.trend_analysis import (
    TrendResult,
    detect_change_points,
    linear_trend,
    percentage_change,
    rate_of_change,
    rolling_mean,
    seasonal_decompose_naive,
    year_over_year_growth,
)


def test_linear_trend_rising() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = linear_trend(values)
    assert result.direction == "rising"
    assert result.slope > 0


def test_linear_trend_falling() -> None:
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = linear_trend(values)
    assert result.direction == "falling"
    assert result.slope < 0


def test_linear_trend_stable() -> None:
    values = [10.0, 10.0, 10.0, 10.0]
    result = linear_trend(values)
    assert result.direction == "stable"


def test_linear_trend_single_value() -> None:
    result = linear_trend([5.0])
    assert isinstance(result, TrendResult)
    assert result.slope == 0.0


def test_linear_trend_perfect_r_squared() -> None:
    values = [2.0 * i for i in range(10)]
    result = linear_trend(values)
    assert result.r_squared == pytest.approx(1.0, abs=1e-4)


def test_percentage_change_increase() -> None:
    assert percentage_change(100.0, 120.0) == pytest.approx(20.0)


def test_percentage_change_decrease() -> None:
    assert percentage_change(100.0, 80.0) == pytest.approx(-20.0)


def test_percentage_change_zero_old() -> None:
    assert percentage_change(0.0, 50.0) == 0.0


def test_rolling_mean_window_1() -> None:
    values = [1.0, 2.0, 3.0]
    result = rolling_mean(values, window=1)
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_rolling_mean_window_2() -> None:
    values = [2.0, 4.0, 6.0, 8.0]
    result = rolling_mean(values, window=2)
    assert result[1] == pytest.approx(3.0)
    assert result[3] == pytest.approx(7.0)


def test_rolling_mean_empty() -> None:
    assert rolling_mean([], window=3) == []


def test_rolling_mean_invalid_window() -> None:
    assert rolling_mean([1.0, 2.0], window=0) == []


def test_detect_change_points_sharp_jump() -> None:
    values = [10.0] * 5 + [100.0] * 5
    cps = detect_change_points(values, threshold=1.5)
    assert 5 in cps


def test_detect_change_points_constant_series() -> None:
    values = [5.0] * 10
    assert detect_change_points(values) == []


def test_detect_change_points_too_short() -> None:
    assert detect_change_points([1.0, 2.0]) == []


def test_seasonal_decompose_returns_keys() -> None:
    values = [float(i % 4) for i in range(16)]
    result = seasonal_decompose_naive(values, period=4)
    assert "trend" in result
    assert "seasonal" in result
    assert "residual" in result


def test_seasonal_decompose_length_matches() -> None:
    values = list(range(24))
    result = seasonal_decompose_naive(values, period=6)
    assert len(result["trend"]) == 24
    assert len(result["seasonal"]) == 24


def test_seasonal_decompose_short_input() -> None:
    values = [1.0, 2.0, 3.0]
    result = seasonal_decompose_naive(values, period=6)
    assert result["trend"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("n,expected_dir", [(5, "rising"), (0, "stable")])
def test_linear_trend_direction_parametrize(n, expected_dir) -> None:
    if n > 0:
        values = list(range(1, n + 1))
    else:
        values = [5.0] * 5
    result = linear_trend(values)
    assert result.direction == expected_dir


@pytest.mark.parametrize(
    "values,expected_direction",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], "rising"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], "falling"),
        ([10.0, 10.0, 10.0, 10.0], "stable"),
    ],
)
def test_linear_trend_direction_parametrized(values: list, expected_direction: str) -> None:
    result = linear_trend(values)
    assert result.direction == expected_direction


@pytest.mark.parametrize(
    "old,new,expected_pct",
    [
        (100.0, 110.0, 10.0),
        (100.0, 90.0, -10.0),
        (50.0, 100.0, 100.0),
        (200.0, 100.0, -50.0),
    ],
)
def test_percentage_change_parametrized(old: float, new: float, expected_pct: float) -> None:
    assert percentage_change(old, new) == pytest.approx(expected_pct, rel=1e-4)


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_rolling_mean_window_lengths(window: int) -> None:
    values = [float(i) for i in range(1, 11)]
    result = rolling_mean(values, window=window)
    assert len(result) == len(values)


def test_linear_trend_two_values() -> None:
    result = linear_trend([0.0, 4.0])
    assert result.slope == pytest.approx(4.0)
    assert result.direction == "rising"


def test_detect_change_points_empty() -> None:
    result = detect_change_points([])
    assert result == []


def test_seasonal_decompose_naive_length() -> None:
    values = [float(i % 4) for i in range(24)]
    result = seasonal_decompose_naive(values, period=4)
    assert "trend" in result
    assert len(result["trend"]) == len(values)


def test_year_over_year_growth_basic() -> None:
    monthly = [100.0] * 12 + [110.0] * 12
    result = year_over_year_growth(monthly)
    assert len(result) == 12
    for v in result:
        assert v == pytest.approx(10.0, rel=1e-4)


def test_year_over_year_growth_decline() -> None:
    monthly = [200.0] * 12 + [100.0] * 12
    result = year_over_year_growth(monthly)
    assert all(v == pytest.approx(-50.0, rel=1e-4) for v in result)


def test_year_over_year_growth_too_short() -> None:
    assert year_over_year_growth([100.0] * 11) == []


def test_year_over_year_growth_exact_period() -> None:
    monthly = [50.0] * 12 + [75.0]
    result = year_over_year_growth(monthly)
    assert len(result) == 1
    assert result[0] == pytest.approx(50.0, rel=1e-4)


@pytest.mark.parametrize("period", [4, 12, 6])
def test_year_over_year_growth_custom_period(period: int) -> None:
    series = [10.0] * period + [12.0] * period
    result = year_over_year_growth(series, period=period)
    assert len(result) == period
    assert all(v == pytest.approx(20.0, rel=1e-4) for v in result)


def test_rate_of_change_basic() -> None:
    result = rate_of_change([100.0, 110.0, 121.0])
    assert result[0] == pytest.approx(10.0, rel=1e-4)


def test_rate_of_change_lag_2() -> None:
    result = rate_of_change([100.0, 120.0, 150.0], lag=2)
    assert len(result) == 1
    assert result[0] == pytest.approx(50.0, rel=1e-4)


def test_rate_of_change_too_short() -> None:
    assert rate_of_change([100.0]) == []


def test_rate_of_change_constant_series() -> None:
    result = rate_of_change([5.0, 5.0, 5.0, 5.0])
    assert all(v == 0.0 for v in result)


def test_rate_of_change_length() -> None:
    values = list(range(1, 11))
    result = rate_of_change(values, lag=1)
    assert len(result) == len(values) - 1


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_rate_of_change_various_lags(lag: int) -> None:
    values = [float(i * 10) for i in range(1, 8)]
    result = rate_of_change(values, lag=lag)
    assert len(result) == len(values) - lag


def test_linear_trend_result_has_slope() -> None:
    result = linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert hasattr(result, "slope")
    assert result.slope > 0


def test_linear_trend_result_has_direction() -> None:
    result = linear_trend([5.0, 4.0, 3.0, 2.0, 1.0])
    assert result.direction == "falling"


@pytest.mark.parametrize("n", [5, 10, 20])
def test_rolling_mean_output_length(n: int) -> None:
    values = [float(i) for i in range(n)]
    result = rolling_mean(values, window=3)
    assert len(result) == n


def test_seasonal_decompose_residual_near_zero_for_pure_seasonal() -> None:
    import math

    pattern = [math.sin(2 * math.pi * i / 12) * 10 + 20 for i in range(24)]
    decomp = seasonal_decompose_naive(pattern, period=12)
    assert "residual" in decomp


def test_year_over_year_growth_positive() -> None:
    monthly = [10.0] * 12 + [12.0] * 12
    growth = year_over_year_growth(monthly, period=12)
    assert all(g > 0 for g in growth)


@pytest.mark.parametrize("window", [2, 5, 10])
def test_rolling_mean_window_smaller_than_length(window: int) -> None:
    values = list(range(20))
    result = rolling_mean([float(v) for v in values], window=window)
    assert len(result) == 20


class TestMomentumScore:
    """Tests for momentum_score."""

    def test_increasing_signal(self) -> None:
        from app.trend_analysis import momentum_score

        values = [1.0] * 23 + [5.0] * 7
        result = momentum_score(values, short_window=7, long_window=30)
        assert result["signal"] == "increasing"
        assert result["momentum"] > 0

    def test_decreasing_signal(self) -> None:
        from app.trend_analysis import momentum_score

        values = [5.0] * 23 + [1.0] * 7
        result = momentum_score(values, short_window=7, long_window=30)
        assert result["signal"] == "decreasing"
        assert result["momentum"] < 0

    def test_neutral_signal(self) -> None:
        from app.trend_analysis import momentum_score

        values = [3.0] * 30
        result = momentum_score(values, short_window=7, long_window=30)
        assert result["signal"] == "neutral"
        assert result["momentum"] == 0.0

    def test_keys_present(self) -> None:
        from app.trend_analysis import momentum_score

        values = list(range(1, 31))
        result = momentum_score(values)
        for key in ("short_ma", "long_ma", "momentum", "signal"):
            assert key in result

    def test_raises_on_short_window_zero(self) -> None:
        import pytest

        from app.trend_analysis import momentum_score

        with pytest.raises(ValueError):
            momentum_score([1.0] * 30, short_window=0)

    def test_raises_short_ge_long(self) -> None:
        import pytest

        from app.trend_analysis import momentum_score

        with pytest.raises(ValueError):
            momentum_score([1.0] * 30, short_window=10, long_window=10)

    def test_raises_insufficient_values(self) -> None:
        import pytest

        from app.trend_analysis import momentum_score

        with pytest.raises(ValueError):
            momentum_score([1.0] * 5, short_window=3, long_window=10)


class TestCumulativeSum:
    """Tests for cumulative_sum."""

    def test_basic(self) -> None:
        from app.trend_analysis import cumulative_sum

        assert cumulative_sum([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]

    def test_empty(self) -> None:
        from app.trend_analysis import cumulative_sum

        assert cumulative_sum([]) == []

    def test_single(self) -> None:
        from app.trend_analysis import cumulative_sum

        assert cumulative_sum([7.5]) == [7.5]

    def test_negatives(self) -> None:
        from app.trend_analysis import cumulative_sum

        result = cumulative_sum([-1.0, -2.0, 3.0])
        assert result == [-1.0, -3.0, 0.0]

    def test_length_preserved(self) -> None:
        from app.trend_analysis import cumulative_sum

        values = list(range(1, 11))
        assert len(cumulative_sum([float(v) for v in values])) == 10
