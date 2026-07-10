"""Tests for multi-step forecaster with confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

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
