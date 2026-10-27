"""Extended tests for quake-net app/forecasting.py Omori law aftershock modeling."""

from __future__ import annotations

import pytest

from app.forecasting import (
    BATH_DELTA,
    DEFAULT_C,
    DEFAULT_P,
    expected_count,
    fit_omori,
    omori_rate,
    productivity_from_magnitude,
)


class TestOmoriRate:
    def test_positive_rate(self) -> None:
        rate = omori_rate(1.0, k=10.0)
        assert rate > 0.0

    def test_rate_decreases_with_time(self) -> None:
        rate_early = omori_rate(0.1, k=10.0)
        rate_late = omori_rate(10.0, k=10.0)
        assert rate_early > rate_late

    def test_negative_time_clamped_to_zero(self) -> None:
        rate_neg = omori_rate(-5.0, k=10.0)
        rate_zero = omori_rate(0.0, k=10.0)
        assert rate_neg == pytest.approx(rate_zero)

    def test_larger_k_gives_higher_rate(self) -> None:
        r1 = omori_rate(1.0, k=5.0)
        r2 = omori_rate(1.0, k=10.0)
        assert r2 > r1

    def test_returns_float(self) -> None:
        assert isinstance(omori_rate(1.0, k=10.0), float)


class TestExpectedCount:
    def test_inverted_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_count(5.0, 1.0, k=10.0)

    def test_zero_interval_near_zero(self) -> None:
        count = expected_count(1.0, 1.0, k=10.0)
        assert count == pytest.approx(0.0, abs=1e-6)

    def test_longer_interval_more_events(self) -> None:
        c1 = expected_count(0.0, 1.0, k=10.0)
        c2 = expected_count(0.0, 10.0, k=10.0)
        assert c2 > c1

    def test_p_equals_one_uses_log_form(self) -> None:
        count = expected_count(0.0, 10.0, k=5.0, p=1.0)
        assert count > 0.0

    def test_non_negative_count(self) -> None:
        count = expected_count(0.0, 5.0, k=10.0)
        assert count >= 0.0


class TestProductivityFromMagnitude:
    def test_m4_gives_unit_productivity(self) -> None:
        result = productivity_from_magnitude(4.0)
        assert result == pytest.approx(1.0)

    def test_larger_magnitude_higher_productivity(self) -> None:
        k5 = productivity_from_magnitude(5.0)
        k6 = productivity_from_magnitude(6.0)
        assert k6 > k5

    def test_positive_result(self) -> None:
        assert productivity_from_magnitude(5.0) > 0.0


class TestFitOmori:
    def test_too_few_events_returns_defaults(self) -> None:
        result = fit_omori([0.1, 0.5])
        assert result["fitted"] is False
        assert result["p"] == pytest.approx(DEFAULT_P)

    def test_enough_events_returns_fitted(self) -> None:
        times = [0.1, 0.3, 0.7, 1.5, 3.0, 7.0, 14.0, 30.0]
        result = fit_omori(times)
        assert result["fitted"] is True

    def test_returns_dict_with_required_keys(self) -> None:
        result = fit_omori([0.1])
        for key in ("k", "p", "c", "n_events", "log_likelihood"):
            assert key in result

    def test_negative_times_excluded(self) -> None:
        result = fit_omori([-1.0, -0.5, 0.1])
        assert result["n_events"] == 1

    def test_k_positive(self) -> None:
        result = fit_omori([0.1, 0.5, 1.0, 2.0, 5.0])
        assert result["k"] > 0.0


class TestConstants:
    def test_bath_delta_positive(self) -> None:
        assert BATH_DELTA > 0.0

    def test_default_p_in_range(self) -> None:
        assert 0.5 < DEFAULT_P < 2.0

    def test_default_c_positive(self) -> None:
        assert DEFAULT_C > 0.0
