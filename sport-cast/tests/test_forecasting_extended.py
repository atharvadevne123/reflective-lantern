"""Extended tests for sport-cast app/forecasting.py."""

from __future__ import annotations

import pytest

from app.forecasting import detect_form_trend, linear_trend, rolling_average


class TestLinearTrend:
    def test_returns_dict(self) -> None:
        result = linear_trend([1.0, 2.0, 3.0])
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        result = linear_trend([1.0, 2.0, 3.0])
        assert "slope" in result
        assert "intercept" in result
        assert "next_predicted" in result

    def test_increasing_series_positive_slope(self) -> None:
        result = linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["slope"] > 0

    def test_decreasing_series_negative_slope(self) -> None:
        result = linear_trend([5.0, 4.0, 3.0, 2.0, 1.0])
        assert result["slope"] < 0

    def test_flat_series_zero_slope(self) -> None:
        result = linear_trend([3.0] * 5)
        assert abs(result["slope"]) < 0.01

    def test_single_value_no_error(self) -> None:
        result = linear_trend([5.0])
        assert result["slope"] == pytest.approx(0.0)

    def test_empty_list_no_error(self) -> None:
        result = linear_trend([])
        assert result["slope"] == pytest.approx(0.0)


class TestRollingAverage:
    def test_empty_returns_empty(self) -> None:
        assert rolling_average([]) == []

    def test_single_value_returns_single(self) -> None:
        result = rolling_average([5.0])
        assert result == [pytest.approx(5.0)]

    def test_length_matches_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_average(values, window=3)
        assert len(result) == len(values)

    def test_first_value_is_input(self) -> None:
        result = rolling_average([7.0, 8.0, 9.0], window=5)
        assert result[0] == pytest.approx(7.0)

    def test_full_window_average(self) -> None:
        result = rolling_average([1.0, 2.0, 3.0, 4.0, 5.0], window=5)
        assert result[-1] == pytest.approx(3.0)


class TestDetectFormTrend:
    def test_improving_trend(self) -> None:
        rates = [0.3, 0.5, 0.7, 0.8, 0.9]
        assert detect_form_trend(rates) == "improving"

    def test_declining_trend(self) -> None:
        rates = [0.9, 0.8, 0.6, 0.4, 0.2]
        assert detect_form_trend(rates) == "declining"

    def test_stable_trend(self) -> None:
        rates = [0.5, 0.52, 0.49, 0.51, 0.50]
        assert detect_form_trend(rates) == "stable"

    def test_too_short_returns_stable(self) -> None:
        assert detect_form_trend([0.8, 0.9]) == "stable"

    def test_empty_returns_stable(self) -> None:
        assert detect_form_trend([]) == "stable"
