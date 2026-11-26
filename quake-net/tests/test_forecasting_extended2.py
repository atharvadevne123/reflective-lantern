"""Extended Omori law forecasting tests with parametrize and boundary coverage."""

from __future__ import annotations

import math

import pytest

from app.forecasting import (
    DEFAULT_C,
    decay_half_life,
    expected_count,
    forecast_sequence,
    moving_average,
    omori_rate,
    productivity_from_magnitude,
)


@pytest.mark.parametrize(
    "t_days,k,expected_gt",
    [
        (0.0, 10.0, 0.0),
        (1.0, 10.0, 0.0),
        (10.0, 100.0, 0.0),
    ],
)
def test_omori_rate_positive(t_days: float, k: float, expected_gt: float) -> None:
    rate = omori_rate(t_days, k=k)
    assert rate > expected_gt


def test_omori_rate_decreases_over_time() -> None:
    k = 50.0
    rates = [omori_rate(float(t), k=k) for t in range(10)]
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1]


def test_omori_rate_negative_time_clamped_to_zero() -> None:
    r_neg = omori_rate(-5.0, k=10.0)
    r_zero = omori_rate(0.0, k=10.0)
    assert r_neg == pytest.approx(r_zero, rel=1e-6)


@pytest.mark.parametrize("p", [0.8, 1.0, 1.1, 1.3, 1.5])
def test_omori_rate_custom_p(p: float) -> None:
    rate = omori_rate(1.0, k=10.0, p=p)
    expected = 10.0 / math.pow(1.0 + DEFAULT_C, p)
    assert rate == pytest.approx(expected, rel=1e-5)


def test_expected_count_positive_interval() -> None:
    count = expected_count(0.0, 1.0, k=10.0)
    assert count > 0.0


def test_expected_count_inverted_interval_raises() -> None:
    with pytest.raises(ValueError):
        expected_count(5.0, 2.0, k=10.0)


def test_expected_count_zero_interval() -> None:
    count = expected_count(2.0, 2.0, k=10.0)
    assert count == pytest.approx(0.0, abs=1e-9)


def test_expected_count_p_equals_1_log_case() -> None:
    count = expected_count(0.0, 10.0, k=5.0, p=1.0)
    assert count > 0.0
    assert math.isfinite(count)


@pytest.mark.parametrize(
    "mag,expected_gt",
    [
        (3.0, 0.0),
        (5.0, 0.0),
        (7.0, 0.0),
        (9.0, 0.0),
    ],
)
def test_productivity_from_magnitude_positive(mag: float, expected_gt: float) -> None:
    k = productivity_from_magnitude(mag)
    assert k > expected_gt


def test_productivity_from_magnitude_increases_with_mag() -> None:
    mags = [4.0, 5.0, 6.0, 7.0, 8.0]
    ks = [productivity_from_magnitude(m) for m in mags]
    for i in range(len(ks) - 1):
        assert ks[i] < ks[i + 1]


@pytest.mark.parametrize("horizon", [1, 3, 7, 14, 30])
def test_forecast_sequence_horizon_length(horizon: int) -> None:
    result = forecast_sequence(6.5, horizon_days=horizon)
    assert len(result["daily_forecast"]) == horizon


def test_forecast_sequence_negative_horizon_raises() -> None:
    with pytest.raises(ValueError):
        forecast_sequence(6.5, horizon_days=-1)


def test_forecast_sequence_zero_horizon_raises() -> None:
    with pytest.raises(ValueError):
        forecast_sequence(6.5, horizon_days=0)


def test_forecast_sequence_total_matches_sum() -> None:
    result = forecast_sequence(5.0, horizon_days=7)
    total = sum(d["expected_count"] for d in result["daily_forecast"])
    assert total == pytest.approx(result["total_expected"], rel=1e-3)


def test_forecast_sequence_bath_law() -> None:
    for mag in [5.0, 6.5, 7.8]:
        result = forecast_sequence(mag, horizon_days=7)
        expected_largest = round(max(0.0, mag - 1.2), 2)
        assert result["largest_expected_aftershock"] == pytest.approx(expected_largest, abs=0.01)


def test_forecast_sequence_with_observed_times() -> None:
    times = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    result = forecast_sequence(6.0, horizon_days=7, observed_times=times)
    assert result["omori_parameters"]["fitted"] is True


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_moving_average_output_length(window: int) -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = moving_average(values, window=window)
    assert len(out) == len(values)


def test_moving_average_window_one_equals_input() -> None:
    values = [1.5, 2.5, 3.5]
    out = moving_average(values, window=1)
    assert out == pytest.approx(values, rel=1e-6)


def test_moving_average_empty_input() -> None:
    assert moving_average([], window=3) == []


def test_moving_average_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        moving_average([1.0, 2.0], window=0)


def test_decay_half_life_positive() -> None:
    hl = decay_half_life(k=10.0)
    assert hl > 0.0


def test_decay_half_life_increases_with_k() -> None:
    hl1 = decay_half_life(k=5.0)
    hl2 = decay_half_life(k=50.0)
    assert hl2 > hl1
