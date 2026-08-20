"""Tests for Omori-law aftershock sequence forecasting."""

from __future__ import annotations

import math

import pytest

from app.forecasting import (
    BATH_DELTA,
    decay_half_life,
    expected_count,
    fit_omori,
    forecast_sequence,
    moving_average,
    omori_rate,
    productivity_from_magnitude,
)


class TestOmoriRate:
    def test_rate_is_positive(self) -> None:
        assert omori_rate(1.0, k=10.0) > 0

    def test_rate_decays_with_time(self) -> None:
        early = omori_rate(0.5, k=10.0)
        late = omori_rate(10.0, k=10.0)
        assert late < early

    def test_no_singularity_at_zero(self) -> None:
        assert math.isfinite(omori_rate(0.0, k=10.0))

    def test_negative_time_clamped(self) -> None:
        assert omori_rate(-5.0, k=10.0) == omori_rate(0.0, k=10.0)

    def test_rate_scales_linearly_with_k(self) -> None:
        assert omori_rate(1.0, k=20.0) == pytest.approx(2 * omori_rate(1.0, k=10.0))

    @pytest.mark.parametrize("p", [0.8, 1.0, 1.2, 1.5])
    def test_all_exponents_finite(self, p: float) -> None:
        assert math.isfinite(omori_rate(2.0, k=5.0, p=p))


class TestExpectedCount:
    def test_count_is_positive(self) -> None:
        assert expected_count(0.0, 7.0, k=10.0) > 0

    def test_longer_window_gives_more_events(self) -> None:
        short = expected_count(0.0, 1.0, k=10.0)
        long = expected_count(0.0, 10.0, k=10.0)
        assert long > short

    def test_later_window_gives_fewer_events(self) -> None:
        first_day = expected_count(0.0, 1.0, k=10.0)
        tenth_day = expected_count(9.0, 10.0, k=10.0)
        assert tenth_day < first_day

    def test_zero_width_window_is_zero(self) -> None:
        assert expected_count(3.0, 3.0, k=10.0) == pytest.approx(0.0, abs=1e-9)

    def test_inverted_window_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            expected_count(5.0, 1.0, k=10.0)

    def test_p_equals_one_uses_log_branch(self) -> None:
        result = expected_count(0.0, 5.0, k=10.0, p=1.0)
        assert math.isfinite(result)
        assert result > 0

    def test_additivity_over_subintervals(self) -> None:
        whole = expected_count(0.0, 6.0, k=10.0)
        parts = expected_count(0.0, 3.0, k=10.0) + expected_count(3.0, 6.0, k=10.0)
        assert whole == pytest.approx(parts, rel=1e-9)


class TestProductivityFromMagnitude:
    def test_larger_magnitude_more_productive(self) -> None:
        assert productivity_from_magnitude(7.0) > productivity_from_magnitude(5.0)

    def test_reference_magnitude_gives_unity(self) -> None:
        assert productivity_from_magnitude(4.0) == pytest.approx(1.0)

    def test_always_positive(self) -> None:
        assert productivity_from_magnitude(1.0) > 0


class TestFitOmori:
    def test_too_few_events_falls_back(self) -> None:
        result = fit_omori([0.1, 0.5])
        assert result["fitted"] is False

    def test_empty_sequence_falls_back(self) -> None:
        assert fit_omori([])["fitted"] is False

    def test_fits_a_real_sequence(self) -> None:
        times = [0.01, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0]
        result = fit_omori(times)
        assert result["fitted"] is True
        assert result["n_events"] == 8

    def test_fitted_p_within_grid(self) -> None:
        times = [0.02, 0.08, 0.2, 0.6, 1.2, 2.5, 5.0]
        assert result_p_in_grid(fit_omori(times)["p"])

    def test_fitted_k_positive(self) -> None:
        times = [0.02, 0.08, 0.2, 0.6, 1.2, 2.5, 5.0]
        assert fit_omori(times)["k"] > 0

    def test_negative_times_ignored(self) -> None:
        result = fit_omori([-1.0, 0.1, 0.4, 1.0, 2.0])
        assert result["n_events"] == 4

    def test_more_events_gives_larger_k(self) -> None:
        sparse = fit_omori([0.1, 0.5, 1.0, 2.0, 4.0])
        dense = fit_omori([0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
        assert dense["k"] > sparse["k"]


def result_p_in_grid(p: float) -> bool:
    return p in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)


class TestForecastSequence:
    def test_returns_one_entry_per_day(self) -> None:
        result = forecast_sequence(6.5, horizon_days=7)
        assert len(result["daily_forecast"]) == 7

    def test_days_are_numbered_from_one(self) -> None:
        result = forecast_sequence(6.0, horizon_days=3)
        assert [d["day"] for d in result["daily_forecast"]] == [1, 2, 3]

    def test_counts_decline_over_horizon(self) -> None:
        daily = forecast_sequence(6.5, horizon_days=10)["daily_forecast"]
        assert daily[-1]["expected_count"] < daily[0]["expected_count"]

    def test_probabilities_are_valid(self) -> None:
        daily = forecast_sequence(7.0, horizon_days=5)["daily_forecast"]
        assert all(0.0 <= d["probability_at_least_one"] <= 1.0 for d in daily)

    def test_bath_law_applied(self) -> None:
        result = forecast_sequence(7.0)
        assert result["largest_expected_aftershock"] == pytest.approx(7.0 - BATH_DELTA)

    def test_largest_aftershock_never_negative(self) -> None:
        assert forecast_sequence(0.5)["largest_expected_aftershock"] >= 0.0

    def test_bigger_mainshock_more_aftershocks(self) -> None:
        small = forecast_sequence(5.0, horizon_days=7)["total_expected"]
        large = forecast_sequence(7.5, horizon_days=7)["total_expected"]
        assert large > small

    def test_observed_times_drive_the_fit(self) -> None:
        result = forecast_sequence(6.0, horizon_days=5, observed_times=[0.1, 0.3, 0.9, 2.0, 4.0])
        assert result["omori_parameters"]["fitted"] is True

    def test_zero_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            forecast_sequence(6.0, horizon_days=0)

    def test_total_matches_daily_sum(self) -> None:
        result = forecast_sequence(6.0, horizon_days=6)
        assert result["total_expected"] == pytest.approx(
            sum(d["expected_count"] for d in result["daily_forecast"]), abs=1e-3
        )


class TestDecayHalfLife:
    def test_half_life_positive(self) -> None:
        assert decay_half_life(k=10.0) > 0

    def test_rate_actually_halves(self) -> None:
        k = 10.0
        t_half = decay_half_life(k=k)
        assert omori_rate(t_half, k=k) == pytest.approx(omori_rate(0.0, k=k) / 2, rel=1e-6)

    def test_zero_k_returns_zero(self) -> None:
        assert decay_half_life(k=0.0) == 0.0


class TestMovingAverage:
    def test_length_preserved(self) -> None:
        assert len(moving_average([1.0, 2.0, 3.0, 4.0], window=2)) == 4

    def test_empty_input(self) -> None:
        assert moving_average([]) == []

    def test_window_of_one_is_identity(self) -> None:
        values = [1.0, 5.0, 3.0]
        assert moving_average(values, window=1) == values

    def test_smooths_a_spike(self) -> None:
        smoothed = moving_average([1.0, 1.0, 10.0, 1.0], window=3)
        assert smoothed[2] < 10.0

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            moving_average([1.0, 2.0], window=0)
