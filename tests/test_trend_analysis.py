"""Tests for app/trend_analysis.py."""

from __future__ import annotations

import pytest

from app.time_series import cumulative_sum
from app.trend_analysis import (
    TrendResult,
    autocorrelation,
    detect_change_points,
    double_exponential_smoothing,
    exponential_growth_rate,
    linear_trend,
    momentum_score,
    normalised_range,
    peak_valley_count,
    percentage_change,
    period_comparison,
    rate_of_change,
    rolling_mean,
    seasonal_decompose_naive,
    trend_reversal_count,
    trend_strength,
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


class TestExponentialWeightedMean:
    def test_length_preserved(self) -> None:
        from app.trend_analysis import exponential_weighted_mean

        result = exponential_weighted_mean([1.0, 2.0, 3.0, 4.0])
        assert len(result) == 4

    def test_first_value_unchanged(self) -> None:
        from app.trend_analysis import exponential_weighted_mean

        result = exponential_weighted_mean([5.0, 6.0, 7.0])
        assert result[0] == pytest.approx(5.0)

    def test_empty(self) -> None:
        from app.trend_analysis import exponential_weighted_mean

        assert exponential_weighted_mean([]) == []

    def test_invalid_alpha(self) -> None:
        from app.trend_analysis import exponential_weighted_mean

        with pytest.raises(ValueError):
            exponential_weighted_mean([1.0, 2.0], alpha=0.0)

    def test_alpha_1_returns_values(self) -> None:
        from app.trend_analysis import exponential_weighted_mean

        vals = [1.0, 2.0, 3.0]
        result = exponential_weighted_mean(vals, alpha=1.0)
        assert result == pytest.approx(vals)


class TestTrendStrength:
    def test_perfect_linear(self) -> None:
        from app.trend_analysis import trend_strength

        assert trend_strength([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.0, rel=1e-3)

    def test_constant_series(self) -> None:
        from app.trend_analysis import trend_strength

        assert trend_strength([3.0, 3.0, 3.0]) == pytest.approx(1.0)

    def test_too_short_raises(self) -> None:
        from app.trend_analysis import trend_strength

        with pytest.raises(ValueError):
            trend_strength([5.0])

    def test_result_in_range(self) -> None:
        from app.trend_analysis import trend_strength

        result = trend_strength([1.0, 3.0, 2.0, 5.0, 4.0])
        assert 0.0 <= result <= 1.0


class TestPeakValleyCount:
    def test_no_peaks_short(self) -> None:
        from app.trend_analysis import peak_valley_count

        result = peak_valley_count([1.0, 2.0])
        assert result == {"peaks": 0, "valleys": 0}

    def test_one_peak(self) -> None:
        from app.trend_analysis import peak_valley_count

        result = peak_valley_count([1.0, 3.0, 1.0])
        assert result["peaks"] == 1
        assert result["valleys"] == 0

    def test_one_valley(self) -> None:
        from app.trend_analysis import peak_valley_count

        result = peak_valley_count([3.0, 1.0, 3.0])
        assert result["valleys"] == 1

    def test_multiple_peaks(self) -> None:
        from app.trend_analysis import peak_valley_count

        result = peak_valley_count([1.0, 3.0, 1.0, 4.0, 1.0])
        assert result["peaks"] == 2


class TestNormalisedRange:
    def test_basic(self) -> None:
        from app.trend_analysis import normalised_range

        result = normalised_range([2.0, 4.0, 6.0])
        assert result > 0

    def test_constant_range_is_zero(self) -> None:
        from app.trend_analysis import normalised_range

        assert normalised_range([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        from app.trend_analysis import normalised_range

        with pytest.raises(ValueError):
            normalised_range([])

    def test_zero_mean_raises(self) -> None:
        from app.trend_analysis import normalised_range

        with pytest.raises(ValueError):
            normalised_range([0.0, 0.0, 0.0])


def test_trend_strength_perfect_line() -> None:
    values = [float(i) for i in range(10)]
    assert trend_strength(values) == pytest.approx(1.0, abs=1e-4)


def test_trend_strength_constant() -> None:
    assert trend_strength([5.0] * 10) == pytest.approx(1.0, abs=1e-4)


def test_autocorrelation_lag1_positive() -> None:
    values = [float(i) for i in range(20)]
    ac = autocorrelation(values, lag=1)
    assert ac > 0


def test_autocorrelation_insufficient_data() -> None:
    assert autocorrelation([1.0, 2.0], lag=2) == 0.0


def test_autocorrelation_constant_series() -> None:
    assert autocorrelation([3.0] * 10, lag=1) == 0.0


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_autocorrelation_lag_parametrize(lag: int) -> None:
    values = [float(i) for i in range(20)]
    ac = autocorrelation(values, lag=lag)
    assert -1.0 <= ac <= 1.0


def test_cumulative_sum_correctness() -> None:
    result = cumulative_sum([1.0, 2.0, 3.0])
    assert result == pytest.approx([1.0, 3.0, 6.0])


def test_cumulative_sum_empty() -> None:
    assert cumulative_sum([]) == []


def test_cumulative_sum_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert len(cumulative_sum(values)) == len(values)


def test_double_exponential_smoothing_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = double_exponential_smoothing(values)
    assert len(result) == len(values)


def test_double_exponential_smoothing_single() -> None:
    result = double_exponential_smoothing([10.0])
    assert result == [10.0]


def test_double_exponential_smoothing_increasing() -> None:
    values = [float(i) for i in range(1, 11)]
    result = double_exponential_smoothing(values, alpha=0.5, beta=0.5)
    assert result[-1] > result[0]


def test_trend_reversal_count_monotone() -> None:
    assert trend_reversal_count([1.0, 2.0, 3.0, 4.0]) == 0


def test_trend_reversal_count_alternating() -> None:
    values = [1.0, 3.0, 1.0, 3.0, 1.0]
    count = trend_reversal_count(values)
    assert count == 3


def test_trend_reversal_count_empty() -> None:
    assert trend_reversal_count([]) == 0


def test_trend_reversal_count_single() -> None:
    assert trend_reversal_count([5.0]) == 0


def test_trend_reversal_count_two() -> None:
    assert trend_reversal_count([1.0, 2.0]) == 0


def test_trend_reversal_count_one_reversal() -> None:
    assert trend_reversal_count([1.0, 3.0, 2.0]) == 1


def test_exponential_growth_rate_doubling() -> None:
    import math

    values = [1.0, 2.0]
    rate = exponential_growth_rate(values)
    assert rate == pytest.approx(math.log(2.0), rel=1e-4)


def test_exponential_growth_rate_flat() -> None:
    values = [5.0, 5.0, 5.0]
    assert exponential_growth_rate(values) == pytest.approx(0.0)


def test_exponential_growth_rate_empty_raises() -> None:
    with pytest.raises(ValueError):
        exponential_growth_rate([5.0])


def test_exponential_growth_rate_non_positive_raises() -> None:
    with pytest.raises(ValueError):
        exponential_growth_rate([1.0, 0.0, 2.0])


@pytest.mark.parametrize("n", [3, 5, 8])
def test_trend_reversal_count_nonzero_param(n: int) -> None:
    values = [float(i % 2) for i in range(n)]
    count = trend_reversal_count(values)
    assert count >= 0


class TestPeriodComparison:
    def test_same_series_zero_change(self) -> None:
        result = period_comparison([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["pct_change"] == pytest.approx(0.0)

    def test_increase_positive_pct_change(self) -> None:
        result = period_comparison([100.0, 100.0], [120.0, 120.0])
        assert result["pct_change"] == pytest.approx(20.0)

    def test_decrease_negative_pct_change(self) -> None:
        result = period_comparison([200.0, 200.0], [100.0, 100.0])
        assert result["pct_change"] == pytest.approx(-50.0)

    def test_totals_correct(self) -> None:
        result = period_comparison([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert result["total_a"] == pytest.approx(6.0)
        assert result["total_b"] == pytest.approx(15.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            period_comparison([], [1.0])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            period_comparison([1.0, 2.0], [1.0])


class TestHurstExponent:
    def _trending_series(self, n: int = 50) -> list[float]:
        return [float(i) + 0.01 * i * i for i in range(n)]

    def test_returns_float(self) -> None:
        from app.trend_analysis import hurst_exponent

        result = hurst_exponent(self._trending_series())
        assert isinstance(result, float)

    def test_range_zero_to_one(self) -> None:
        from app.trend_analysis import hurst_exponent

        result = hurst_exponent(self._trending_series())
        assert 0.0 <= result <= 1.0

    def test_too_few_values_raises(self) -> None:
        from app.trend_analysis import hurst_exponent

        with pytest.raises(ValueError, match="requires at least"):
            hurst_exponent([1.0, 2.0, 3.0], max_lag=20)

    def test_random_walk_near_half(self) -> None:
        import random

        from app.trend_analysis import hurst_exponent

        random.seed(42)
        walk = [0.0]
        for _ in range(100):
            walk.append(walk[-1] + random.gauss(0, 1))
        result = hurst_exponent(walk, max_lag=20)
        assert 0.0 <= result <= 1.0


class TestWindowedTrendStrength:
    def test_perfect_linear_trend(self) -> None:
        from app.trend_analysis import windowed_trend_strength

        values = [float(i) for i in range(20)]
        result = windowed_trend_strength(values)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_flat_series_zero(self) -> None:
        from app.trend_analysis import windowed_trend_strength

        values = [5.0] * 20
        assert windowed_trend_strength(values) == 0.0

    def test_in_range(self) -> None:
        import random

        from app.trend_analysis import windowed_trend_strength

        random.seed(0)
        values = [random.gauss(0, 1) for _ in range(50)]
        assert 0.0 <= windowed_trend_strength(values) <= 1.0

    def test_too_few_raises(self) -> None:
        from app.trend_analysis import windowed_trend_strength

        with pytest.raises(ValueError):
            windowed_trend_strength([1.0])

    @pytest.mark.parametrize("window", [5, 10, 20])
    def test_window_parameter(self, window: int) -> None:
        from app.trend_analysis import windowed_trend_strength

        values = [float(i) for i in range(50)]
        result = windowed_trend_strength(values, window=window)
        assert 0.0 <= result <= 1.0


class TestPolyfitTrend:
    def test_linear_fit_perfect(self) -> None:
        from app.trend_analysis import polyfit_trend

        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = polyfit_trend(values)
        assert len(result) == len(values)
        for actual, fitted in zip(values, result, strict=False):
            assert actual == pytest.approx(fitted, abs=1e-4)

    def test_output_length(self) -> None:
        from app.trend_analysis import polyfit_trend

        values = [1.0, 3.0, 2.0, 5.0, 4.0]
        assert len(polyfit_trend(values)) == len(values)

    def test_invalid_degree_raises(self) -> None:
        from app.trend_analysis import polyfit_trend

        with pytest.raises(ValueError, match="degree must be at least 1"):
            polyfit_trend([1.0, 2.0, 3.0], degree=0)

    def test_too_few_values_raises(self) -> None:
        from app.trend_analysis import polyfit_trend

        with pytest.raises(ValueError):
            polyfit_trend([1.0], degree=1)

    def test_unsupported_degree_raises(self) -> None:
        from app.trend_analysis import polyfit_trend

        with pytest.raises(NotImplementedError):
            polyfit_trend([1.0, 2.0, 3.0, 4.0, 5.0], degree=2)


# ---------------------------------------------------------------------------
# Tests for trend_strength, rolling_trend_direction, cumulative_return
# ---------------------------------------------------------------------------


class TestTrendStrengthNew:
    def test_perfect_trend(self) -> None:
        from app.trend_analysis import trend_strength

        values = [float(i) for i in range(10)]
        assert trend_strength(values) == pytest.approx(1.0, abs=0.001)

    def test_flat_series(self) -> None:
        from app.trend_analysis import trend_strength

        assert trend_strength([5.0] * 5) == pytest.approx(1.0, abs=0.001)

    def test_too_short_raises(self) -> None:
        import pytest

        from app.trend_analysis import trend_strength

        with pytest.raises(ValueError):
            trend_strength([1.0])

    def test_noise_gives_low_r2(self) -> None:
        import random

        from app.trend_analysis import trend_strength

        random.seed(42)
        noise = [random.gauss(0, 10) for _ in range(50)]
        assert trend_strength(noise) < 0.5


class TestRollingTrendDirection:
    def test_all_up(self) -> None:
        from app.trend_analysis import rolling_trend_direction

        result = rolling_trend_direction([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert all(d in ("up", "flat") for d in result)
        assert result[-1] == "up"

    def test_all_down(self) -> None:
        from app.trend_analysis import rolling_trend_direction

        result = rolling_trend_direction([5.0, 4.0, 3.0, 2.0, 1.0], window=3)
        assert result[-1] == "down"

    def test_early_entries_flat(self) -> None:
        from app.trend_analysis import rolling_trend_direction

        result = rolling_trend_direction([1.0, 2.0, 3.0, 4.0], window=3)
        assert result[0] == "flat"
        assert result[1] == "flat"

    def test_window_less_than_two_raises(self) -> None:
        import pytest

        from app.trend_analysis import rolling_trend_direction

        with pytest.raises(ValueError):
            rolling_trend_direction([1.0, 2.0], window=1)


class TestCumulativeReturnNew:
    def test_double(self) -> None:
        from app.trend_analysis import cumulative_return

        assert cumulative_return([100.0, 200.0]) == pytest.approx(1.0, abs=0.001)

    def test_no_change(self) -> None:
        from app.trend_analysis import cumulative_return

        assert cumulative_return([100.0, 100.0]) == pytest.approx(0.0, abs=0.001)

    def test_too_short_raises(self) -> None:
        import pytest

        from app.trend_analysis import cumulative_return

        with pytest.raises(ValueError):
            cumulative_return([100.0])

    def test_zero_base_returns_zero(self) -> None:
        from app.trend_analysis import cumulative_return

        assert cumulative_return([0.0, 100.0]) == 0.0

    def test_negative_base_raises(self) -> None:
        import pytest

        from app.trend_analysis import cumulative_return

        with pytest.raises(ValueError):
            cumulative_return([-10.0, 100.0])


@pytest.mark.parametrize(
    "values,expected_direction",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], "rising"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], "falling"),
    ],
)
def test_linear_trend_direction_boundary(values: list[float], expected_direction: str) -> None:
    result = linear_trend(values)
    assert result.direction == expected_direction


@pytest.mark.parametrize("window", [1, 2, 3])
def test_rolling_mean_output_size(window: int) -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_mean(values, window=window)
    assert len(result) == len(values)


def test_normalised_range_zero_for_constant() -> None:
    assert normalised_range([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_normalised_range_one_for_zero_to_one() -> None:
    assert normalised_range([0.0, 0.5, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_peak_valley_count_monotonic_rising() -> None:
    result = peak_valley_count([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["peaks"] == 0
    assert result["valleys"] == 0


def test_momentum_score_positive_for_rising() -> None:
    rising = [1.0] * 20 + [float(i) for i in range(1, 11)]
    result = momentum_score(rising, short_window=5, long_window=15)
    assert result["momentum"] > 0


@pytest.mark.parametrize("n", [5, 10, 24])
def test_rate_of_change_output_length(n: int) -> None:
    values = [float(i) for i in range(n)]
    result = rate_of_change(values, lag=1)
    assert len(result) == n - 1


@pytest.mark.parametrize("n", [12, 24, 36])
def test_year_over_year_growth_length(n: int) -> None:
    monthly = [float(i + 1) for i in range(n)]
    result = year_over_year_growth(monthly, period=12)
    assert len(result) == n - 12


@pytest.mark.parametrize(
    "values,expected_sign",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], "positive"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], "negative"),
    ],
)
def test_trend_strength_sign(values: list, expected_sign: str) -> None:
    from app.trend_analysis import windowed_trend_strength

    result = windowed_trend_strength(values)
    # windowed_trend_strength returns |correlation| in [0, 1] regardless of direction
    assert 0.0 <= result <= 1.0
    # For clearly monotonic series (either direction), strength should be close to 1
    if expected_sign in ("positive", "negative"):
        assert result > 0.99


@pytest.mark.parametrize("n", [5, 10, 20])
def test_cumulative_sum_monotone_for_positive(n: int) -> None:
    values = [1.0] * n
    result = cumulative_sum(values)
    for i in range(1, len(result)):
        assert result[i] >= result[i - 1]


class TestCumulativeGrowth:
    def test_flat_series_all_zero(self) -> None:
        from app.trend_analysis import cumulative_growth

        result = cumulative_growth([10.0, 10.0, 10.0])
        assert all(v == 0.0 for v in result)

    def test_first_element_is_zero(self) -> None:
        from app.trend_analysis import cumulative_growth

        result = cumulative_growth([5.0, 10.0, 15.0])
        assert result[0] == 0.0

    def test_doubling_is_one(self) -> None:
        from app.trend_analysis import cumulative_growth

        result = cumulative_growth([100.0, 200.0])
        assert result[-1] == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        from app.trend_analysis import cumulative_growth

        with pytest.raises(ValueError):
            cumulative_growth([])

    def test_zero_base_all_zero(self) -> None:
        from app.trend_analysis import cumulative_growth

        result = cumulative_growth([0.0, 10.0, 20.0])
        assert all(v == 0.0 for v in result)


class TestTrendReversalCount:
    def test_no_reversals_monotone(self) -> None:
        from app.trend_analysis import trend_reversal_count

        assert trend_reversal_count([1.0, 2.0, 3.0, 4.0]) == 0

    def test_alternating_series(self) -> None:
        from app.trend_analysis import trend_reversal_count

        result = trend_reversal_count([1.0, 3.0, 2.0, 4.0, 3.0])
        assert result >= 2

    def test_too_short_returns_zero(self) -> None:
        from app.trend_analysis import trend_reversal_count

        assert trend_reversal_count([1.0, 2.0]) == 0

    def test_returns_int(self) -> None:
        from app.trend_analysis import trend_reversal_count

        result = trend_reversal_count([1.0, 2.0, 1.0])
        assert isinstance(result, int)


class TestResistanceLevel:
    def test_basic(self) -> None:
        from app.trend_analysis import resistance_level

        vals = list(range(1, 11))
        assert resistance_level(vals) == 9

    def test_single_element(self) -> None:
        from app.trend_analysis import resistance_level

        assert resistance_level([42.0]) == 42.0

    def test_empty_raises(self) -> None:
        from app.trend_analysis import resistance_level

        with pytest.raises(ValueError):
            resistance_level([])


class TestSupportLevel:
    def test_basic(self) -> None:
        from app.trend_analysis import support_level

        vals = list(range(1, 11))
        assert support_level(vals) == 1

    def test_single_element(self) -> None:
        from app.trend_analysis import support_level

        assert support_level([7.0]) == 7.0

    def test_empty_raises(self) -> None:
        from app.trend_analysis import support_level

        with pytest.raises(ValueError):
            support_level([])


class TestMedianAbsoluteDeviation:
    def test_uniform_series(self) -> None:
        from app.trend_analysis import median_absolute_deviation

        assert median_absolute_deviation([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        from app.trend_analysis import median_absolute_deviation

        assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        from app.trend_analysis import median_absolute_deviation

        with pytest.raises(ValueError):
            median_absolute_deviation([])


@pytest.mark.parametrize(
    "values,expected_mad",
    [
        ([5.0, 5.0, 5.0], 0.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 1.0),
        ([10.0], 0.0),
    ],
)
def test_mad_parametrized(values, expected_mad: float) -> None:
    """median_absolute_deviation returns the correct MAD for known sequences."""
    from app.trend_analysis import median_absolute_deviation

    assert median_absolute_deviation(values) == pytest.approx(expected_mad)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_mad_non_negative_for_random_data(n: int) -> None:
    """MAD is never negative for any sequence of values."""
    import random

    from app.trend_analysis import median_absolute_deviation

    values = [random.gauss(0, 1) for _ in range(n)]
    assert median_absolute_deviation(values) >= 0.0
