"""Tests for fleet-mind app/predictive_analytics.py."""

from __future__ import annotations

import pytest

from app.predictive_analytics import (
    RULInput,
    linear_degradation_forecast,
    maintenance_urgency_score,
    remaining_useful_life_km,
    time_to_threshold,
    weibull_failure_probability,
)


def _rul(**kwargs) -> RULInput:
    defaults = dict(current_value=8.0, degradation_rate_per_km=0.0001, failure_threshold=2.0)
    defaults.update(kwargs)
    return RULInput(**defaults)


class TestRemainingUsefulLifeKm:
    def test_basic_rul(self) -> None:
        # (8.0 - 2.0) / 0.001 = 6000 km
        inp = _rul(current_value=8.0, degradation_rate_per_km=0.001, failure_threshold=2.0)
        assert remaining_useful_life_km(inp) == pytest.approx(6000.0)

    def test_already_failed_returns_zero(self) -> None:
        inp = _rul(current_value=1.0, failure_threshold=2.0)
        assert remaining_useful_life_km(inp) == pytest.approx(0.0)

    def test_zero_rate_returns_inf(self) -> None:
        inp = _rul(degradation_rate_per_km=0.0)
        assert remaining_useful_life_km(inp) == float("inf")


class TestWeibullFailureProbability:
    def test_zero_age_returns_zero(self) -> None:
        assert weibull_failure_probability(0.0, scale=50_000.0) == pytest.approx(0.0)

    def test_at_scale_roughly_half(self) -> None:
        p = weibull_failure_probability(50_000.0, scale=50_000.0, shape=2.0)
        assert 0.5 < p < 0.7

    def test_increases_with_age(self) -> None:
        p1 = weibull_failure_probability(10_000.0, 50_000.0)
        p2 = weibull_failure_probability(40_000.0, 50_000.0)
        assert p2 > p1


class TestLinearDegradationForecast:
    def test_no_degradation(self) -> None:
        assert linear_degradation_forecast(8.0, 0.0, 10_000.0) == pytest.approx(8.0)

    def test_decreases_with_km(self) -> None:
        result = linear_degradation_forecast(8.0, 0.001, 1000.0)
        assert result == pytest.approx(7.0)


class TestTimeToThreshold:
    def test_already_at_threshold(self) -> None:
        assert time_to_threshold(2.0, 2.0, 0.001) == pytest.approx(0.0)

    def test_calculates_correctly(self) -> None:
        # (8.0 - 2.0) / 0.001 = 6000
        assert time_to_threshold(8.0, 2.0, 0.001) == pytest.approx(6000.0)

    def test_zero_rate_returns_inf(self) -> None:
        assert time_to_threshold(8.0, 2.0, 0.0) == float("inf")


class TestMaintenanceUrgencyScore:
    def test_zero_rul_returns_one(self) -> None:
        assert maintenance_urgency_score(0.0) == pytest.approx(1.0)

    def test_high_rul_returns_low_score(self) -> None:
        assert maintenance_urgency_score(50_000.0) == pytest.approx(0.0)

    def test_between_zero_and_one(self) -> None:
        score = maintenance_urgency_score(25_000.0)
        assert 0.0 <= score <= 1.0
