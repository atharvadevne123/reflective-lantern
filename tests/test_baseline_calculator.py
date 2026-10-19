"""Tests for app.baseline_calculator module."""

from __future__ import annotations

import pytest

from app.baseline_calculator import (
    BaselineModel,
    adjusted_baseline,
    build_regression_baseline,
    energy_savings,
    linear_regression,
    mean_baseline,
    normalised_savings_percent,
)


class TestMeanBaseline:
    def test_uniform_values(self) -> None:
        assert mean_baseline([5.0, 5.0, 5.0]) == pytest.approx(5.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_baseline([])

    def test_single_value(self) -> None:
        assert mean_baseline([42.0]) == pytest.approx(42.0)

    @pytest.mark.parametrize(
        "values,expected",
        [([1.0, 3.0], 2.0), ([10.0, 20.0, 30.0], 20.0), ([0.0, 100.0], 50.0)],
    )
    def test_parametrized(self, values: list[float], expected: float) -> None:
        assert mean_baseline(values) == pytest.approx(expected)


class TestLinearRegression:
    def test_perfect_line(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        a, b, r2 = linear_regression(x, y)
        assert a == pytest.approx(0.0, abs=1e-6)
        assert b == pytest.approx(2.0, rel=1e-5)
        assert r2 == pytest.approx(1.0, rel=1e-5)

    def test_horizontal_line(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0]
        a, b, r2 = linear_regression(x, y)
        assert b == pytest.approx(0.0, abs=1e-6)
        assert r2 == pytest.approx(1.0)

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            linear_regression([1.0, 2.0], [3.0])

    def test_too_few_points_raise(self) -> None:
        with pytest.raises(ValueError):
            linear_regression([1.0], [2.0])

    def test_r_squared_between_zero_one(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 2.0, 5.0]
        _, _, r2 = linear_regression(x, y)
        assert 0.0 <= r2 <= 1.0

    @pytest.mark.parametrize(
        "x,y",
        [
            ([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]),
            ([10.0, 20.0, 30.0], [100.0, 200.0, 300.0]),
        ],
    )
    def test_parametrized_proportional(
        self, x: list[float], y: list[float]
    ) -> None:
        _, _, r2 = linear_regression(x, y)
        assert r2 == pytest.approx(1.0, rel=1e-5)


class TestAdjustedBaseline:
    def test_zero_slope(self) -> None:
        assert adjusted_baseline(100.0, 0.0, 50.0) == pytest.approx(100.0)

    def test_with_slope(self) -> None:
        assert adjusted_baseline(0.0, 2.0, 10.0) == pytest.approx(20.0)

    def test_negative_x_allowed(self) -> None:
        result = adjusted_baseline(50.0, 1.0, -10.0)
        assert result == pytest.approx(40.0)


class TestEnergySavings:
    def test_savings_positive_when_reduced(self) -> None:
        assert energy_savings(100.0, 80.0) == pytest.approx(20.0)

    def test_savings_negative_when_increased(self) -> None:
        assert energy_savings(100.0, 120.0) == pytest.approx(-20.0)

    def test_no_change_zero_savings(self) -> None:
        assert energy_savings(100.0, 100.0) == pytest.approx(0.0)


class TestBuildRegressionBaseline:
    def test_returns_baseline_model(self) -> None:
        x = [100.0, 200.0, 300.0, 400.0]
        y = [500.0, 900.0, 1300.0, 1700.0]
        result = build_regression_baseline(x, y, 250.0)
        assert isinstance(result, BaselineModel)

    def test_method_label(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [2.0, 4.0, 6.0]
        model = build_regression_baseline(x, y, 4.0)
        assert "regression" in model.method.lower()

    def test_baseline_kwh_positive(self) -> None:
        x = [10.0, 20.0, 30.0]
        y = [100.0, 200.0, 300.0]
        model = build_regression_baseline(x, y, 25.0)
        assert model.baseline_kwh > 0.0


class TestNormalisedSavingsPercent:
    def test_full_saving(self) -> None:
        assert normalised_savings_percent(100.0, 0.0) == pytest.approx(100.0)

    def test_no_saving(self) -> None:
        assert normalised_savings_percent(100.0, 100.0) == pytest.approx(0.0)

    def test_increase_negative_percent(self) -> None:
        assert normalised_savings_percent(100.0, 110.0) == pytest.approx(-10.0)

    def test_zero_baseline_returns_zero(self) -> None:
        assert normalised_savings_percent(0.0, 50.0) == 0.0

    def test_negative_baseline_raises(self) -> None:
        with pytest.raises(ValueError):
            normalised_savings_percent(-50.0, 30.0)

    @pytest.mark.parametrize(
        "baseline,actual,expected",
        [(200.0, 160.0, 20.0), (500.0, 250.0, 50.0), (100.0, 100.0, 0.0)],
    )
    def test_parametrized(
        self, baseline: float, actual: float, expected: float
    ) -> None:
        assert normalised_savings_percent(baseline, actual) == pytest.approx(expected)
