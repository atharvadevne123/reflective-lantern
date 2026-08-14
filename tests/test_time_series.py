"""Time-series forecasting utility tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.time_series import (
    cumulative_sum,
    daily_totals,
    detect_spikes,
    exponential_moving_average,
    find_changepoints,
    first_nonzero,
    forecast_linear_trend,
    hampel_filter,
    load_factor,
    mape,
    min_max_scale,
    moving_max,
    moving_median,
    normalize_series,
    pair_difference,
    peak_hours,
    peak_to_valley_ratio,
    percent_change,
    rolling_zscore,
    seasonal_baseline,
    simple_moving_average,
    z_normalize,
)


def test_sma_length() -> None:
    result = simple_moving_average([1.0] * 10, window=3)
    assert len(result) == 10


def test_sma_flat_series() -> None:
    result = simple_moving_average([5.0] * 20, window=5)
    assert all(abs(v - 5.0) < 0.1 for v in result[-10:])


def test_seasonal_baseline_length() -> None:
    data = list(range(48))
    baseline = seasonal_baseline(data, period=24)
    assert len(baseline) == 48


def test_seasonal_baseline_periodicity() -> None:
    data = [float(i % 24) for i in range(72)]
    baseline = seasonal_baseline(data, period=24)
    # positions 0, 24, 48 should all have the same baseline
    assert abs(baseline[0] - baseline[24]) < 1e-9
    assert abs(baseline[0] - baseline[48]) < 1e-9


def test_linear_trend_length() -> None:
    result = forecast_linear_trend([1.0, 2.0, 3.0, 4.0], horizon=10)
    assert len(result) == 10


def test_linear_trend_direction() -> None:
    # Ascending series → future values should be higher than last historical
    result = forecast_linear_trend(list(range(20)), horizon=5)
    assert result[0] > 18.0


def test_detect_spikes_finds_outlier() -> None:
    data = [10.0] * 100
    data[50] = 1000.0
    spikes = detect_spikes(data)
    assert 50 in spikes


def test_detect_spikes_empty_on_flat() -> None:
    data = [5.0] * 50
    assert detect_spikes(data) == []


@pytest.mark.parametrize("window", [1, 3, 7, 24])
def test_sma_various_windows(window) -> None:
    data = list(np.random.default_rng(42).uniform(5, 30, 100))
    result = simple_moving_average(data, window=window)
    assert len(result) == 100


def test_detect_spikes_empty_input() -> None:
    assert detect_spikes([]) == []


def test_detect_spikes_returns_indices() -> None:
    data = [1.0] * 50 + [999.0]
    spikes = detect_spikes(data)
    assert 50 in spikes


def test_peak_hours_returns_top_n() -> None:
    data = [1.0, 5.0, 3.0, 9.0, 2.0]
    peaks = peak_hours(data, top_n=2)
    assert len(peaks) == 2
    assert peaks[0] == 3  # index of 9.0


def test_peak_hours_empty_input() -> None:
    assert peak_hours([]) == []


def test_peak_hours_top_n_larger_than_series() -> None:
    data = [1.0, 2.0]
    assert len(peak_hours(data, top_n=100)) == 2


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_linear_trend_various_horizons(horizon) -> None:
    result = forecast_linear_trend(list(range(30)), horizon=horizon)
    assert len(result) == horizon


def test_cumulative_sum_basic() -> None:
    assert cumulative_sum([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]


def test_cumulative_sum_empty() -> None:
    assert cumulative_sum([]) == []


def test_moving_max_basic() -> None:
    import math

    result = moving_max([1.0, 3.0, 2.0, 5.0], window=3)
    assert math.isnan(result[0])
    assert result[2] == pytest.approx(3.0)
    assert result[3] == pytest.approx(5.0)


def test_moving_max_too_short_all_nan() -> None:
    import math

    result = moving_max([1.0, 2.0], window=5)
    assert all(math.isnan(v) for v in result)


def test_normalize_series_basic() -> None:
    result = normalize_series([0.0, 5.0, 10.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_series_constant() -> None:
    assert normalize_series([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]


def test_daily_totals_default_period() -> None:
    result = daily_totals([1.0] * 48, period=24)
    assert result == [24.0, 24.0]


def test_daily_totals_empty() -> None:
    assert daily_totals([]) == []


def test_first_nonzero_finds_index() -> None:
    assert first_nonzero([0.0, 0.0, 3.0]) == 2


def test_first_nonzero_all_zeros() -> None:
    assert first_nonzero([0.0, 0.0]) == -1


def test_ema_first_value_equals_input() -> None:
    data = [5.0, 10.0, 15.0]
    result = exponential_moving_average(data, alpha=0.5)
    assert result[0] == pytest.approx(5.0)


def test_ema_alpha_one_equals_input() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = exponential_moving_average(data, alpha=1.0)
    assert result == pytest.approx(data)


def test_ema_empty_input() -> None:
    assert exponential_moving_average([], alpha=0.3) == []


def test_ema_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        exponential_moving_average([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        exponential_moving_average([1.0, 2.0], alpha=1.5)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9, 1.0])
def test_ema_various_alphas(alpha) -> None:
    data = [float(i) for i in range(20)]
    result = exponential_moving_average(data, alpha=alpha)
    assert len(result) == 20


def test_forecast_trend_with_seasonality_length() -> None:
    from app.time_series import forecast_trend_with_seasonality

    data = [float(i % 24) for i in range(72)]
    result = forecast_trend_with_seasonality(data, horizon=12, period=24)
    assert len(result) == 12


def test_forecast_trend_with_seasonality_non_negative() -> None:
    from app.time_series import forecast_trend_with_seasonality

    data = [10.0 + i * 0.1 for i in range(48)]
    result = forecast_trend_with_seasonality(data, horizon=24, period=24)
    assert all(v >= 0 for v in result)


def test_forecast_trend_with_seasonality_empty_input() -> None:
    from app.time_series import forecast_trend_with_seasonality

    assert forecast_trend_with_seasonality([], horizon=10) == []


def test_forecast_trend_with_seasonality_zero_horizon() -> None:
    from app.time_series import forecast_trend_with_seasonality

    data = list(range(48))
    assert forecast_trend_with_seasonality(data, horizon=0) == []


@pytest.mark.parametrize("horizon,period", [(6, 12), (24, 24), (48, 24), (7, 7)])
def test_forecast_trend_with_seasonality_parametrized(horizon, period) -> None:
    from app.time_series import forecast_trend_with_seasonality

    data = [float(i % period) * 2 + 5 for i in range(4 * period)]
    result = forecast_trend_with_seasonality(data, horizon=horizon, period=period)
    assert len(result) == horizon


def test_moving_range_basic() -> None:
    from app.time_series import moving_range

    result = moving_range([1.0, 3.0, 2.0, 5.0])
    assert result == pytest.approx([2.0, 1.0, 3.0])


def test_moving_range_empty_input() -> None:
    from app.time_series import moving_range

    assert moving_range([]) == []


def test_moving_range_single_element() -> None:
    from app.time_series import moving_range

    assert moving_range([5.0]) == []


def test_moving_range_flat_series() -> None:
    from app.time_series import moving_range

    result = moving_range([4.0] * 10)
    assert all(v == pytest.approx(0.0) for v in result)


def test_moving_range_length() -> None:
    from app.time_series import moving_range

    data = list(range(20))
    assert len(moving_range(data)) == 19


def test_consumption_variance_flat() -> None:
    from app.time_series import consumption_variance

    assert consumption_variance([5.0] * 10) == pytest.approx(0.0)


def test_consumption_variance_two_values() -> None:
    from app.time_series import consumption_variance

    result = consumption_variance([0.0, 2.0])
    assert result == pytest.approx(1.0)


def test_consumption_variance_empty() -> None:
    from app.time_series import consumption_variance

    assert consumption_variance([]) == 0.0


def test_consumption_variance_single() -> None:
    from app.time_series import consumption_variance

    assert consumption_variance([7.0]) == 0.0


@pytest.mark.parametrize(
    "data,expected_len",
    [
        ([1.0, 2.0, 3.0], 2),
        ([10.0] * 5, 4),
    ],
)
def test_moving_range_parametrized_length(data, expected_len) -> None:
    from app.time_series import moving_range

    assert len(moving_range(data)) == expected_len


# --- New tests for recently added functions ---


def test_moving_range_basic_v2() -> None:
    from app.time_series import moving_range

    values = [1.0, 3.0, 2.0, 5.0]
    result = moving_range(values)
    assert result == [2.0, 1.0, 3.0]


def test_moving_range_too_short_v2() -> None:
    from app.time_series import moving_range

    assert moving_range([5.0]) == []
    assert moving_range([]) == []


def test_consumption_variance_flat_v2() -> None:
    from app.time_series import consumption_variance

    result = consumption_variance([3.0] * 10)
    assert result == 0.0


def test_consumption_variance_known_v2() -> None:
    from app.time_series import consumption_variance

    result = consumption_variance([2.0, 4.0])
    assert abs(result - 1.0) < 1e-9


def test_consumption_variance_too_short_v2() -> None:
    from app.time_series import consumption_variance

    assert consumption_variance([5.0]) == 0.0


def test_forecast_trend_with_seasonality_length_v2() -> None:
    from app.time_series import forecast_trend_with_seasonality

    values = [float(i % 24) for i in range(48)]
    result = forecast_trend_with_seasonality(values, horizon=12, period=24)
    assert len(result) == 12


def test_forecast_trend_with_seasonality_empty_v2() -> None:
    from app.time_series import forecast_trend_with_seasonality

    assert forecast_trend_with_seasonality([], horizon=5) == []


def test_forecast_trend_with_seasonality_non_negative_v2() -> None:
    from app.time_series import forecast_trend_with_seasonality

    values = [max(0, float(i) + 5) for i in range(48)]
    result = forecast_trend_with_seasonality(values, horizon=12)
    assert all(v >= 0 for v in result)


def test_resample_hourly_to_daily_full_day_v2() -> None:
    from app.time_series import resample_hourly_to_daily

    hourly = [1.0] * 48
    result = resample_hourly_to_daily(hourly)
    assert len(result) == 2
    assert all(abs(v - 24.0) < 1e-9 for v in result)


def test_resample_hourly_to_daily_partial_v2() -> None:
    from app.time_series import resample_hourly_to_daily

    hourly = [1.0] * 25
    result = resample_hourly_to_daily(hourly)
    assert len(result) == 2


def test_resample_hourly_to_daily_empty_v2() -> None:
    from app.time_series import resample_hourly_to_daily

    assert resample_hourly_to_daily([]) == []


def test_cumulative_consumption_basic_v2() -> None:
    from app.time_series import cumulative_consumption

    result = cumulative_consumption([1.0, 2.0, 3.0])
    assert abs(result[-1] - 6.0) < 1e-6


def test_cumulative_consumption_empty_v2() -> None:
    from app.time_series import cumulative_consumption

    assert cumulative_consumption([]) == []


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_ema_same_length(alpha) -> None:
    values = [float(i) for i in range(20)]
    from app.time_series import exponential_moving_average

    result = exponential_moving_average(values, alpha=alpha)
    assert len(result) == len(values)


def test_detect_plateau_flat_series() -> None:
    from app.time_series import detect_plateau

    values = [10.0] * 8
    plateaus = detect_plateau(values, tolerance=0.1)
    assert len(plateaus) >= 1
    assert plateaus[0][0] == 0
    assert plateaus[0][1] == 7


def test_detect_plateau_no_plateau() -> None:
    from app.time_series import detect_plateau

    values = [1.0, 5.0, 1.0, 5.0, 1.0]
    plateaus = detect_plateau(values, tolerance=0.1)
    assert len(plateaus) == 0


def test_detect_plateau_empty() -> None:
    from app.time_series import detect_plateau

    assert detect_plateau([]) == []


def test_clip_outliers_basic() -> None:
    from app.time_series import clip_outliers

    values = [1.0] * 8 + [1000.0]
    clipped = clip_outliers(values, upper_pct=90.0)
    assert max(clipped) < 1000.0


def test_clip_outliers_preserves_length() -> None:
    from app.time_series import clip_outliers

    values = list(range(20))
    clipped = clip_outliers([float(v) for v in values])
    assert len(clipped) == 20


def test_clip_outliers_empty() -> None:
    from app.time_series import clip_outliers

    assert clip_outliers([]) == []


@pytest.mark.parametrize("upper_pct", [75.0, 90.0, 99.0])
def test_clip_outliers_various_percentiles(upper_pct) -> None:
    from app.time_series import clip_outliers

    values = list(range(1, 101))
    clipped = clip_outliers([float(v) for v in values], upper_pct=upper_pct)
    assert max(clipped) <= upper_pct + 1


@pytest.mark.parametrize("window", [1, 3, 5, 10])
def test_sma_window_length(window: int) -> None:
    values = [float(i) for i in range(20)]
    result = simple_moving_average(values, window=window)
    assert len(result) == len(values)


@pytest.mark.parametrize("period", [4, 7, 12, 24])
def test_seasonal_baseline_various_periods(period: int) -> None:
    data = [float(i % period) for i in range(period * 4)]
    baseline = seasonal_baseline(data, period=period)
    assert len(baseline) == len(data)
    # Verify periodicity
    assert abs(baseline[0] - baseline[period]) < 1e-9


@pytest.mark.parametrize("horizon", [1, 5, 10, 24])
def test_forecast_linear_trend_horizons(horizon: int) -> None:
    history = [float(i) for i in range(10)]
    result = forecast_linear_trend(history, horizon=horizon)
    assert len(result) == horizon


def test_detect_spikes_returns_index() -> None:
    values = [5.0] * 20 + [100.0] + [5.0] * 20
    spikes = detect_spikes(values, z_threshold=3.0)
    assert 20 in spikes


def test_detect_spikes_no_spikes() -> None:
    values = [10.0 + i * 0.1 for i in range(20)]
    spikes = detect_spikes(values, z_threshold=10.0)
    assert len(spikes) == 0


@pytest.mark.parametrize("n_peaks", [3, 5, 10])
def test_peak_hours_returns_correct_count(n_peaks: int) -> None:
    values = [float(i) for i in range(24)]
    peaks = peak_hours(values, top_n=n_peaks)
    assert len(peaks) == min(n_peaks, 24)


def test_find_changepoints_stable_series_no_cps() -> None:
    values = [10.0] * 50
    assert find_changepoints(values) == []


def test_find_changepoints_too_short() -> None:
    assert find_changepoints([1.0, 2.0]) == []


def test_find_changepoints_obvious_shift() -> None:
    values = [1.0] * 30 + [100.0] * 30
    cps = find_changepoints(values, min_segment_len=5)
    assert len(cps) >= 1


def test_rolling_zscore_length() -> None:
    values = [float(i) for i in range(20)]
    result = rolling_zscore(values, window=5)
    assert len(result) == len(values)


def test_rolling_zscore_first_is_zero() -> None:
    result = rolling_zscore([10.0, 20.0, 30.0])
    assert result[0] == 0.0


def test_rolling_zscore_empty() -> None:
    assert rolling_zscore([]) == []


def test_rolling_zscore_constant_series() -> None:
    result = rolling_zscore([5.0] * 10)
    assert all(v == 0.0 for v in result)


@pytest.mark.parametrize("window", [3, 5, 10, 24])
def test_rolling_zscore_various_windows(window: int) -> None:
    values = [float(i % 10) for i in range(50)]
    result = rolling_zscore(values, window=window)
    assert len(result) == len(values)


def test_load_factor_uniform() -> None:
    result = load_factor([5.0, 5.0, 5.0, 5.0])
    assert result == pytest.approx(1.0)


def test_load_factor_basic() -> None:
    result = load_factor([1.0, 2.0, 3.0, 4.0])
    assert result == pytest.approx(0.625)


def test_load_factor_empty() -> None:
    assert load_factor([]) == 0.0


def test_load_factor_all_zeros() -> None:
    assert load_factor([0.0, 0.0, 0.0]) == 0.0


def test_peak_to_valley_ratio_basic() -> None:
    result = peak_to_valley_ratio([2.0, 4.0, 6.0, 8.0])
    assert result == pytest.approx(4.0)


def test_peak_to_valley_ratio_empty() -> None:
    assert peak_to_valley_ratio([]) == 0.0


def test_peak_to_valley_ratio_zero_valley() -> None:
    assert peak_to_valley_ratio([0.0, 5.0, 10.0]) == 0.0


def test_peak_to_valley_ratio_constant() -> None:
    assert peak_to_valley_ratio([3.0, 3.0, 3.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values,expected_lf",
    [
        ([10.0, 10.0], 1.0),
        ([0.0, 10.0], 0.5),
    ],
)
def test_load_factor_parametrized(values: list, expected_lf: float) -> None:
    assert load_factor(values) == pytest.approx(expected_lf, rel=1e-4)


def test_moving_median_constant_series() -> None:
    result = moving_median([5.0, 5.0, 5.0, 5.0], window=3)
    assert all(v == pytest.approx(5.0) for v in result)


def test_moving_median_length_preserved() -> None:
    values = [1.0, 3.0, 2.0, 4.0, 5.0]
    result = moving_median(values, window=3)
    assert len(result) == len(values)


def test_moving_median_empty() -> None:
    assert moving_median([], window=3) == []


def test_moving_median_window_1_equals_input() -> None:
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    result = moving_median(values, window=1)
    assert result == pytest.approx(values)


@pytest.mark.parametrize("window", [1, 3, 5])
def test_moving_median_various_windows(window: int) -> None:
    values = [float(i) for i in range(10)]
    result = moving_median(values, window=window)
    assert len(result) == len(values)


def test_clip_outliers_returns_same_length() -> None:
    from app.time_series import clip_outliers

    values = [1.0, 2.0, 100.0, 3.0, 4.0]
    result = clip_outliers(values)
    assert len(result) == len(values)


def test_clip_outliers_reduces_extreme() -> None:
    from app.time_series import clip_outliers

    values = [1.0, 2.0, 3.0, 4.0, 1000.0]
    result = clip_outliers(values, upper_pct=95.0)
    assert result[-1] <= 1000.0


def test_load_factor_returns_float() -> None:
    from app.time_series import load_factor

    result = load_factor([10.0, 20.0, 30.0])
    assert isinstance(result, float)


def test_load_factor_constant_input_is_one() -> None:
    from app.time_series import load_factor

    result = load_factor([15.0] * 10)
    assert result == pytest.approx(1.0)


def test_peak_to_valley_ratio_greater_than_one() -> None:
    from app.time_series import peak_to_valley_ratio

    result = peak_to_valley_ratio([1.0, 5.0, 2.0, 8.0, 3.0])
    assert result >= 1.0


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0],
        [10.0, 10.0, 10.0],
        [5.0, 3.0, 8.0, 2.0],
    ],
)
def test_load_factor_various_inputs(values) -> None:
    from app.time_series import load_factor

    result = load_factor(values)
    assert 0.0 <= result <= 1.0


class TestAutocorrelation:
    def test_perfect_correlation_at_lag0_like(self) -> None:
        from app.time_series import autocorrelation

        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = autocorrelation(vals, lag=1)
        assert isinstance(result, float)

    def test_returns_value_between_minus1_and_1(self) -> None:
        from app.time_series import autocorrelation

        vals = [float(i % 3) for i in range(24)]
        result = autocorrelation(vals, lag=3)
        assert -1.0 <= result <= 1.0

    def test_constant_series_returns_zero(self) -> None:
        from app.time_series import autocorrelation

        assert autocorrelation([5.0] * 10, lag=1) == 0.0

    def test_too_short_returns_zero(self) -> None:
        from app.time_series import autocorrelation

        assert autocorrelation([1.0, 2.0], lag=3) == 0.0

    def test_invalid_lag_raises(self) -> None:
        import pytest

        from app.time_series import autocorrelation

        with pytest.raises(ValueError):
            autocorrelation([1.0, 2.0, 3.0], lag=0)

    @pytest.mark.parametrize("lag", [1, 2, 4, 8])
    def test_various_lags(self, lag: int) -> None:
        from app.time_series import autocorrelation

        vals = [float(i) for i in range(50)]
        result = autocorrelation(vals, lag=lag)
        assert -1.0 <= result <= 1.0


class TestHoltWintersSmooth:
    def test_same_length_as_input(self) -> None:
        from app.time_series import holt_winters_smooth

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(holt_winters_smooth(vals)) == len(vals)

    def test_flat_series_stays_flat(self) -> None:
        from app.time_series import holt_winters_smooth

        vals = [5.0] * 10
        result = holt_winters_smooth(vals)
        assert all(abs(r - 5.0) < 0.5 for r in result)

    def test_too_few_elements_raises(self) -> None:
        import pytest

        from app.time_series import holt_winters_smooth

        with pytest.raises(ValueError):
            holt_winters_smooth([1.0])

    def test_invalid_alpha_raises(self) -> None:
        import pytest

        from app.time_series import holt_winters_smooth

        with pytest.raises(ValueError):
            holt_winters_smooth([1.0, 2.0, 3.0], alpha=0.0)

    @pytest.mark.parametrize("alpha,beta", [(0.1, 0.1), (0.5, 0.3), (0.9, 0.5)])
    def test_various_params_return_list(self, alpha: float, beta: float) -> None:
        from app.time_series import holt_winters_smooth

        vals = [float(i) for i in range(10)]
        result = holt_winters_smooth(vals, alpha=alpha, beta=beta)
        assert len(result) == len(vals)


class TestDetectTrendReversal:
    def test_basic_returns_list(self) -> None:
        from app.time_series import detect_trend_reversal

        vals = list(range(20))
        result = detect_trend_reversal(vals, window=4)
        assert len(result) == 20

    def test_values_binary(self) -> None:
        from app.time_series import detect_trend_reversal

        vals = list(range(20))
        result = detect_trend_reversal(vals, window=4)
        assert all(v in (0, 1) for v in result)

    def test_too_short_all_zero(self) -> None:
        from app.time_series import detect_trend_reversal

        result = detect_trend_reversal([1.0, 2.0, 3.0], window=5)
        assert all(v == 0 for v in result)

    def test_window_one_raises(self) -> None:
        from app.time_series import detect_trend_reversal

        with pytest.raises(ValueError, match="at least 2"):
            detect_trend_reversal([1.0, 2.0, 3.0, 4.0], window=1)

    def test_v_shape_detects_reversal(self) -> None:
        from app.time_series import detect_trend_reversal

        vals = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = detect_trend_reversal(vals, window=3)
        assert any(v == 1 for v in result)


class TestMonthlyTotals:
    def test_basic(self) -> None:
        from app.time_series import monthly_totals

        daily = [1.0] * 360
        result = monthly_totals(daily, [30] * 12)
        assert len(result) == 12
        assert all(v == pytest.approx(30.0) for v in result)

    def test_too_few_days_raises(self) -> None:
        from app.time_series import monthly_totals

        with pytest.raises(ValueError, match="exceeds"):
            monthly_totals([1.0] * 10, [30] * 12)

    def test_default_12_months(self) -> None:
        from app.time_series import monthly_totals

        daily = [2.0] * 360
        result = monthly_totals(daily)
        assert len(result) == 12

    @pytest.mark.parametrize("n_months", [1, 3, 6])
    def test_sum_equals_input_sum(self, n_months: int) -> None:
        from app.time_series import monthly_totals

        daily = list(range(30 * n_months))
        result = monthly_totals(daily, [30] * n_months)
        assert sum(result) == pytest.approx(sum(daily), rel=1e-5)


class TestSeasonalVariance:
    def test_constant_series_zero_variance(self) -> None:
        from app.time_series import seasonal_variance

        result = seasonal_variance([5.0] * 48, period=24)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_empty_raises(self) -> None:
        from app.time_series import seasonal_variance

        with pytest.raises(ValueError, match="empty"):
            seasonal_variance([], period=24)

    def test_period_zero_raises(self) -> None:
        from app.time_series import seasonal_variance

        with pytest.raises(ValueError, match="at least 1"):
            seasonal_variance([1.0, 2.0, 3.0], period=0)

    def test_positive_variance(self) -> None:
        from app.time_series import seasonal_variance

        vals = [float(i % 10) for i in range(100)]
        assert seasonal_variance(vals, period=10) >= 0.0


class TestTrailingAverage:
    def test_basic(self) -> None:
        from app.time_series import trailing_average

        result = trailing_average([1.0, 2.0, 3.0, 4.0, 5.0], n=3)
        assert len(result) == 5
        assert result[2] == pytest.approx(2.0, rel=1e-4)

    def test_n1_equals_input(self) -> None:
        from app.time_series import trailing_average

        vals = [1.0, 5.0, 3.0, 7.0]
        assert trailing_average(vals, n=1) == vals

    def test_empty_raises(self) -> None:
        from app.time_series import trailing_average

        with pytest.raises(ValueError, match="empty"):
            trailing_average([], n=3)

    def test_n_zero_raises(self) -> None:
        from app.time_series import trailing_average

        with pytest.raises(ValueError, match="at least 1"):
            trailing_average([1.0, 2.0], n=0)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_output_length(self, n: int) -> None:
        from app.time_series import trailing_average

        vals = list(range(1, 11))
        assert len(trailing_average(vals, n=n)) == 10


class TestDetectOutlierWindows:
    def test_no_outliers(self) -> None:
        from app.time_series import detect_outlier_windows

        vals = [1.0] * 48
        assert detect_outlier_windows(vals, window=24) == []

    def test_empty_raises(self) -> None:
        from app.time_series import detect_outlier_windows

        with pytest.raises(ValueError, match="empty"):
            detect_outlier_windows([], window=24)

    def test_window_zero_raises(self) -> None:
        from app.time_series import detect_outlier_windows

        with pytest.raises(ValueError, match="at least 1"):
            detect_outlier_windows([1.0, 2.0], window=0)

    def test_spike_window_flagged(self) -> None:
        from app.time_series import detect_outlier_windows

        vals = [1.0] * 24 + [1000.0] * 24 + [1.0] * 24
        result = detect_outlier_windows(vals, window=24, threshold_std=1.0)
        assert len(result) > 0

    def test_returns_tuple_pairs(self) -> None:
        from app.time_series import detect_outlier_windows

        vals = [1.0] * 48 + [999.0] * 24
        result = detect_outlier_windows(vals, window=24, threshold_std=1.0)
        for start, end in result:
            assert start <= end


def test_cumulative_consumption_empty() -> None:
    from app.time_series import cumulative_consumption

    assert cumulative_consumption([]) == []


def test_cumulative_consumption_values() -> None:
    from app.time_series import cumulative_consumption

    result = cumulative_consumption([1.0, 2.0, 3.0])
    assert result == pytest.approx([1.0, 3.0, 6.0])


def test_cumulative_consumption_monotone() -> None:
    from app.time_series import cumulative_consumption

    data = [0.5, 1.5, 2.0, 0.1]
    result = cumulative_consumption(data)
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_resample_hourly_to_daily_exact() -> None:
    from app.time_series import resample_hourly_to_daily

    data = [1.0] * 48  # 2 full days
    result = resample_hourly_to_daily(data)
    assert result == pytest.approx([24.0, 24.0])


def test_resample_hourly_to_daily_partial_day() -> None:
    from app.time_series import resample_hourly_to_daily

    data = [1.0] * 25  # 1 day + 1 hour
    result = resample_hourly_to_daily(data)
    assert len(result) == 2
    assert result[0] == pytest.approx(24.0)
    assert result[1] == pytest.approx(1.0)


def test_resample_hourly_to_daily_empty() -> None:
    from app.time_series import resample_hourly_to_daily

    assert resample_hourly_to_daily([]) == []


def test_ema_length() -> None:
    result = exponential_moving_average([1.0] * 10, alpha=0.3)
    assert len(result) == 10


def test_z_normalize_mean_zero() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = z_normalize(values)
    assert abs(sum(result)) < 1e-9


def test_z_normalize_std_one() -> None:
    import math

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = z_normalize(values)
    mean = sum(result) / len(result)
    variance = sum((v - mean) ** 2 for v in result) / len(result)
    assert math.sqrt(variance) == pytest.approx(1.0, abs=1e-4)


def test_z_normalize_constant_series() -> None:
    result = z_normalize([5.0] * 10)
    assert all(v == 0.0 for v in result)


def test_z_normalize_empty() -> None:
    assert z_normalize([]) == []


def test_mape_basic() -> None:
    actual = [100.0, 200.0, 300.0]
    predicted = [110.0, 190.0, 300.0]
    result = mape(actual, predicted)
    assert result == pytest.approx((10.0 + 5.0 + 0.0) / 3, rel=1e-4)


def test_mape_zero_actual_skipped() -> None:
    actual = [0.0, 100.0]
    predicted = [50.0, 110.0]
    result = mape(actual, predicted)
    assert result == pytest.approx(10.0)


def test_mape_empty() -> None:
    assert mape([], []) == 0.0


def test_hampel_filter_length_preserved() -> None:
    values = [1.0, 2.0, 100.0, 2.0, 1.0]
    result = hampel_filter(values)
    assert len(result) == len(values)


def test_hampel_filter_replaces_outlier() -> None:
    values = [1.0, 2.0, 1.0, 1.0, 100.0, 1.0, 1.0, 2.0, 1.0]
    result = hampel_filter(values, window=3, n_sigma=2.0)
    assert result[4] < 10.0


def test_hampel_filter_empty() -> None:
    assert hampel_filter([]) == []


def test_min_max_scale_range() -> None:
    values = [0.0, 5.0, 10.0]
    result = min_max_scale(values)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_min_max_scale_custom_range() -> None:
    values = [0.0, 10.0]
    result = min_max_scale(values, feature_range=(-1.0, 1.0))
    assert result[0] == pytest.approx(-1.0)
    assert result[-1] == pytest.approx(1.0)


def test_min_max_scale_constant() -> None:
    result = min_max_scale([5.0] * 5)
    assert all(v == pytest.approx(0.0) for v in result)


def test_min_max_scale_empty() -> None:
    assert min_max_scale([]) == []


@pytest.mark.parametrize(
    "values,expected_min,expected_max",
    [
        ([0.0, 10.0, 5.0], 0.0, 1.0),
        ([3.0, 6.0, 9.0], 0.0, 1.0),
    ],
)
def test_min_max_scale_parametrize(values, expected_min, expected_max) -> None:
    result = min_max_scale(values)
    assert min(result) == pytest.approx(expected_min)
    assert max(result) == pytest.approx(expected_max)


def test_percent_change_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    result = percent_change(values)
    assert len(result) == len(values) - 1


def test_percent_change_doubling() -> None:
    result = percent_change([1.0, 2.0])
    assert result[0] == pytest.approx(100.0)


def test_percent_change_halving() -> None:
    result = percent_change([2.0, 1.0])
    assert result[0] == pytest.approx(-50.0)


def test_percent_change_empty_returns_empty() -> None:
    assert percent_change([]) == []


def test_percent_change_single_returns_empty() -> None:
    assert percent_change([5.0]) == []


def test_percent_change_zero_denominator_raises() -> None:
    with pytest.raises(ValueError):
        percent_change([0.0, 1.0])


def test_pair_difference_basic() -> None:
    result = pair_difference([5.0, 7.0, 9.0], [1.0, 2.0, 3.0])
    assert result == pytest.approx([4.0, 5.0, 6.0])


def test_pair_difference_zero_diff() -> None:
    result = pair_difference([1.0, 2.0], [1.0, 2.0])
    assert all(v == pytest.approx(0.0) for v in result)


def test_pair_difference_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        pair_difference([1.0, 2.0], [1.0])


def test_pair_difference_empty() -> None:
    assert pair_difference([], []) == []


@pytest.mark.parametrize("n", [3, 5, 10])
def test_pair_difference_length_param(n: int) -> None:
    a = [float(i) for i in range(n)]
    b = [0.0] * n
    result = pair_difference(a, b)
    assert len(result) == n


class TestMovingPercentile:
    def test_median_constant_series(self) -> None:
        from app.time_series import moving_percentile

        result = moving_percentile([5.0, 5.0, 5.0, 5.0], window=3, percentile=50.0)
        assert all(v == pytest.approx(5.0) for v in result)

    def test_same_length_as_input(self) -> None:
        from app.time_series import moving_percentile

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_percentile(values, window=3)
        assert len(result) == len(values)

    def test_100th_percentile_is_max(self) -> None:
        from app.time_series import moving_percentile

        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = moving_percentile(values, window=3, percentile=100.0)
        assert result[2] == pytest.approx(4.0)

    def test_0th_percentile_is_min(self) -> None:
        from app.time_series import moving_percentile

        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = moving_percentile(values, window=3, percentile=0.0)
        assert result[2] == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        from app.time_series import moving_percentile

        with pytest.raises(ValueError):
            moving_percentile([], window=3)

    def test_invalid_window_raises(self) -> None:
        from app.time_series import moving_percentile

        with pytest.raises(ValueError):
            moving_percentile([1.0, 2.0], window=0)

    def test_invalid_percentile_raises(self) -> None:
        from app.time_series import moving_percentile

        with pytest.raises(ValueError):
            moving_percentile([1.0, 2.0], window=2, percentile=101.0)


class TestDoubleExponentialSmoothing:
    def test_output_length(self) -> None:
        from app.time_series import double_exponential_smoothing

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = double_exponential_smoothing(values)
        assert len(result) == len(values)

    def test_too_few_raises(self) -> None:
        from app.time_series import double_exponential_smoothing

        with pytest.raises(ValueError, match="at least 2"):
            double_exponential_smoothing([1.0])

    def test_invalid_alpha_raises(self) -> None:
        from app.time_series import double_exponential_smoothing

        with pytest.raises(ValueError, match="alpha"):
            double_exponential_smoothing([1.0, 2.0, 3.0], alpha=1.5)

    def test_invalid_beta_raises(self) -> None:
        from app.time_series import double_exponential_smoothing

        with pytest.raises(ValueError, match="beta"):
            double_exponential_smoothing([1.0, 2.0, 3.0], beta=0.0)

    def test_upward_trend_tracked(self) -> None:
        from app.time_series import double_exponential_smoothing

        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = double_exponential_smoothing(values, alpha=0.9, beta=0.9)
        assert result[-1] > result[0]

    @pytest.mark.parametrize("alpha,beta", [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])
    def test_returns_float_list(self, alpha: float, beta: float) -> None:
        from app.time_series import double_exponential_smoothing

        result = double_exponential_smoothing([1.0, 2.0, 3.0, 4.0], alpha=alpha, beta=beta)
        assert all(isinstance(v, float) for v in result)


class TestLinearInterpolation:
    def test_no_gaps(self) -> None:
        from app.time_series import linear_interpolation

        values: list[float | None] = [1.0, 2.0, 3.0]
        assert linear_interpolation(values) == [1.0, 2.0, 3.0]

    def test_interior_gap(self) -> None:
        from app.time_series import linear_interpolation

        values: list[float | None] = [0.0, None, 2.0]
        result = linear_interpolation(values)
        assert len(result) == 3
        assert result[1] == pytest.approx(1.0, abs=1e-5)

    def test_leading_gap_filled(self) -> None:
        from app.time_series import linear_interpolation

        values: list[float | None] = [None, None, 5.0]
        result = linear_interpolation(values)
        assert result[0] == result[1] == 5.0

    def test_all_none_raises(self) -> None:
        from app.time_series import linear_interpolation

        with pytest.raises(ValueError, match="non-None"):
            linear_interpolation([None, None, None])

    def test_empty_raises(self) -> None:
        from app.time_series import linear_interpolation

        with pytest.raises(ValueError):
            linear_interpolation([])


class TestForwardFill:
    def test_no_gaps(self) -> None:
        from app.time_series import forward_fill

        values: list[float | None] = [1.0, 2.0, 3.0]
        assert forward_fill(values) == [1.0, 2.0, 3.0]

    def test_trailing_gap(self) -> None:
        from app.time_series import forward_fill

        values: list[float | None] = [1.0, 2.0, None, None]
        result = forward_fill(values)
        assert result == [1.0, 2.0, 2.0, 2.0]

    def test_leading_none_raises(self) -> None:
        from app.time_series import forward_fill

        with pytest.raises(ValueError, match="None"):
            forward_fill([None, 1.0, 2.0])

    def test_empty_raises(self) -> None:
        from app.time_series import forward_fill

        with pytest.raises(ValueError):
            forward_fill([])

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([5.0, None, None], [5.0, 5.0, 5.0]),
            ([1.0, None, 3.0], [1.0, 1.0, 3.0]),
        ],
    )
    def test_parametrized(self, values: list, expected: list) -> None:
        from app.time_series import forward_fill

        assert forward_fill(values) == expected


# ---------------------------------------------------------------------------
# Tests for peak_valley_count, crossings_count, series_range_by_window
# ---------------------------------------------------------------------------


class TestPeakValleyCount:
    def test_single_peak(self) -> None:
        from app.time_series import peak_valley_count

        result = peak_valley_count([1.0, 5.0, 2.0])
        assert result["peaks"] == 1
        assert result["valleys"] == 0

    def test_single_valley(self) -> None:
        from app.time_series import peak_valley_count

        result = peak_valley_count([5.0, 1.0, 5.0])
        assert result["valleys"] == 1
        assert result["peaks"] == 0

    def test_monotonic_series(self) -> None:
        from app.time_series import peak_valley_count

        result = peak_valley_count([1.0, 2.0, 3.0, 4.0])
        assert result["peaks"] == 0 and result["valleys"] == 0

    def test_too_short_raises(self) -> None:
        import pytest

        from app.time_series import peak_valley_count

        with pytest.raises(ValueError):
            peak_valley_count([1.0, 2.0])

    def test_multiple_peaks_and_valleys(self) -> None:
        from app.time_series import peak_valley_count

        result = peak_valley_count([1.0, 3.0, 1.0, 4.0, 2.0])
        assert result["peaks"] >= 2
        assert result["valleys"] >= 1


class TestCrossingsCount:
    def test_no_crossings(self) -> None:
        from app.time_series import crossings_count

        assert crossings_count([1.0, 2.0, 3.0]) == 0

    def test_one_crossing(self) -> None:
        from app.time_series import crossings_count

        assert crossings_count([1.0, -1.0]) == 1

    def test_multiple_crossings(self) -> None:
        from app.time_series import crossings_count

        values = [1.0, -1.0, 1.0, -1.0]
        assert crossings_count(values) == 3

    def test_custom_threshold(self) -> None:
        from app.time_series import crossings_count

        assert crossings_count([0.5, 1.5, 0.5], threshold=1.0) == 2


class TestSeriesRangeByWindow:
    def test_window_one_is_zeros(self) -> None:
        from app.time_series import series_range_by_window

        result = series_range_by_window([3.0, 5.0, 2.0], window=1)
        assert result == [0.0, 0.0, 0.0]

    def test_full_window(self) -> None:
        from app.time_series import series_range_by_window

        result = series_range_by_window([1.0, 3.0, 2.0, 5.0], window=4)
        assert result[-1] == pytest.approx(4.0, abs=0.01)

    def test_window_less_than_one_raises(self) -> None:
        import pytest

        from app.time_series import series_range_by_window

        with pytest.raises(ValueError):
            series_range_by_window([1.0, 2.0], window=0)

    def test_length_preserved(self) -> None:
        from app.time_series import series_range_by_window

        values = [float(i) for i in range(10)]
        assert len(series_range_by_window(values, window=3)) == 10


@pytest.mark.parametrize("window", [2, 5, 10])
def test_simple_moving_average_length(window: int) -> None:
    from app.time_series import simple_moving_average

    values = list(range(20))
    result = simple_moving_average(values, window=window)
    assert len(result) == 20 - window + 1


@pytest.mark.parametrize("horizon", [6, 12, 24])
def test_forecast_linear_trend_length(horizon: int) -> None:
    from app.time_series import forecast_linear_trend

    values = [float(i) for i in range(30)]
    result = forecast_linear_trend(values, horizon=horizon)
    assert len(result) == horizon


@pytest.mark.parametrize(
    "values,expected_load_factor",
    [
        ([1.0, 1.0, 1.0], 1.0),
        ([0.0, 0.0, 10.0], pytest.approx(1 / 3, abs=0.01)),
    ],
)
def test_load_factor_parametrized(values: list, expected_load_factor) -> None:
    from app.time_series import load_factor

    assert load_factor(values) == expected_load_factor


@pytest.mark.parametrize("n", [24, 48, 96])
def test_cumulative_sum_length(n: int) -> None:
    from app.time_series import cumulative_sum

    values = [1.0] * n
    result = cumulative_sum(values)
    assert len(result) == n


@pytest.mark.parametrize(
    "values,alpha",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.1),
        ([10.0, 8.0, 6.0, 4.0], 0.5),
    ],
)
def test_exponential_moving_average_length(values: list, alpha: float) -> None:
    from app.time_series import exponential_moving_average

    result = exponential_moving_average(values, alpha=alpha)
    assert len(result) == len(values)
