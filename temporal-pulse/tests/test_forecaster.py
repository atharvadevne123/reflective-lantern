"""Tests for multi-step forecaster with confidence intervals."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_sine_series(n: int = 100) -> np.ndarray:
    """Return a noisy sine wave series."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 8 * np.pi, n)
    return (np.sin(t) + rng.normal(0, 0.05, n)).astype(np.float32)


class TestPrepareSupervisedData:
    def test_shapes_match(self):
        from app.forecaster import prepare_supervised_data

        series = make_sine_series(100)
        X, y = prepare_supervised_data(series, lookback=20, horizon=5)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 20

    def test_too_short_series_raises(self):
        from app.forecaster import prepare_supervised_data

        with pytest.raises(ValueError):
            prepare_supervised_data(np.array([1.0, 2.0]), lookback=20, horizon=5)

    @pytest.mark.parametrize("lookback,horizon", [(10, 1), (20, 5), (30, 10)])
    def test_various_configs(self, lookback, horizon):
        from app.forecaster import prepare_supervised_data

        series = make_sine_series(200)
        X, y = prepare_supervised_data(series, lookback=lookback, horizon=horizon)
        assert len(X) == 200 - lookback - horizon + 1


class TestFitChannelForecaster:
    def test_returns_model_and_metrics(self):
        from app.forecaster import fit_channel_forecaster

        series = make_sine_series()
        model, metrics = fit_channel_forecaster(series)
        assert model is not None
        assert "mae" in metrics
        assert "mse" in metrics

    def test_fits_sine_wave_well(self):
        from app.forecaster import fit_channel_forecaster

        series = make_sine_series(300)
        _, metrics = fit_channel_forecaster(series)
        assert metrics["mae"] < 0.5


class TestForecastWithConfidence:
    def test_point_and_bounds_lengths(self):
        from app.forecaster import fit_channel_forecaster, forecast_with_confidence

        series = make_sine_series()
        model, _ = fit_channel_forecaster(series, lookback=20)
        fc = forecast_with_confidence(model, series[-20:], steps=5)
        assert len(fc["point"]) == 5
        assert len(fc["lower"]) == 5
        assert len(fc["upper"]) == 5

    def test_bounds_bracket_point(self):
        from app.forecaster import fit_channel_forecaster, forecast_with_confidence

        series = make_sine_series()
        model, _ = fit_channel_forecaster(series, lookback=20)
        fc = forecast_with_confidence(model, series[-20:], steps=3)
        for lo, pt, hi in zip(fc["lower"], fc["point"], fc["upper"]):
            assert lo <= pt <= hi


class TestMultiChannelForecast:
    def test_forecasts_all_channels(self):
        from app.forecaster import multi_channel_forecast

        channels = {"temp": make_sine_series(), "pressure": make_sine_series() * 100}
        results = multi_channel_forecast(channels, horizon=3)
        assert set(results.keys()) == {"temp", "pressure"}

    def test_short_channel_reports_error(self):
        from app.forecaster import multi_channel_forecast

        results = multi_channel_forecast({"tiny": np.array([1.0, 2.0], dtype=np.float32)})
        assert "error" in results["tiny"]


class TestForecastAccuracySummary:
    def test_perfect_forecast_zero_error(self) -> None:
        from app.forecaster import forecast_accuracy_summary

        actual = [1.0, 2.0, 3.0]
        result = forecast_accuracy_summary(actual, actual)
        assert result["mae"] == pytest.approx(0.0, abs=1e-6)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_length_mismatch_raises(self) -> None:
        from app.forecaster import forecast_accuracy_summary

        with pytest.raises(ValueError):
            forecast_accuracy_summary([1.0, 2.0], [1.0])

    def test_empty_raises(self) -> None:
        from app.forecaster import forecast_accuracy_summary

        with pytest.raises(ValueError):
            forecast_accuracy_summary([], [])

    def test_returns_required_keys(self) -> None:
        from app.forecaster import forecast_accuracy_summary

        result = forecast_accuracy_summary([1.0, 2.0, 3.0], [1.1, 1.9, 3.1])
        assert {"mae", "rmse", "mape"} <= result.keys()

    @pytest.mark.parametrize("offset", [0.0, 0.5, 1.0])
    def test_mae_equals_offset(self, offset: float) -> None:
        from app.forecaster import forecast_accuracy_summary

        actual = [10.0, 10.0, 10.0]
        predicted = [10.0 + offset] * 3
        result = forecast_accuracy_summary(actual, predicted)
        assert result["mae"] == pytest.approx(offset, abs=1e-4)


class TestIntervalCoverage:
    def test_full_coverage(self) -> None:
        from app.forecaster import interval_coverage

        assert interval_coverage([5.0, 6.0], [4.0, 5.0], [6.0, 7.0]) == pytest.approx(1.0)

    def test_no_coverage(self) -> None:
        from app.forecaster import interval_coverage

        assert interval_coverage([10.0, 20.0], [0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self) -> None:
        from app.forecaster import interval_coverage

        assert interval_coverage([], [], []) == 0.0
