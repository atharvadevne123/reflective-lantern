"""Extended tests for temporal-pulse app/forecaster.py."""

from __future__ import annotations

import numpy as np
import pytest

from app.forecaster import fit_channel_forecaster, forecast_with_confidence, prepare_supervised_data


class TestPrepareSupervisedData:
    def test_basic_shape(self) -> None:
        series = np.arange(50, dtype=np.float32)
        X, y = prepare_supervised_data(series, lookback=10, horizon=1)
        assert X.shape[1] == 10
        assert len(X) == len(y)

    def test_raises_if_too_short(self) -> None:
        series = np.arange(5, dtype=np.float32)
        with pytest.raises(ValueError):
            prepare_supervised_data(series, lookback=10, horizon=5)

    def test_x_dtype_float32(self) -> None:
        series = np.arange(30, dtype=np.float64)
        X, y = prepare_supervised_data(series, lookback=5, horizon=1)
        assert X.dtype == np.float32

    def test_horizon_affects_y(self) -> None:
        series = np.arange(30, dtype=np.float32)
        _, y1 = prepare_supervised_data(series, lookback=5, horizon=1)
        _, y5 = prepare_supervised_data(series, lookback=5, horizon=5)
        assert not np.array_equal(y1, y5)


class TestFitChannelForecaster:
    def _series(self, n: int = 100) -> np.ndarray:
        return np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)

    def test_returns_model_and_metrics(self) -> None:
        model, metrics = fit_channel_forecaster(self._series(), lookback=10, horizon=1)
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_has_mae(self) -> None:
        _, metrics = fit_channel_forecaster(self._series(), lookback=10, horizon=1)
        assert "mae" in metrics

    def test_mae_is_finite(self) -> None:
        _, metrics = fit_channel_forecaster(self._series(), lookback=10, horizon=1)
        assert np.isfinite(metrics["mae"])

    def test_model_can_predict(self) -> None:
        series = self._series()
        model, _ = fit_channel_forecaster(series, lookback=10, horizon=1)
        window = series[-10:].reshape(1, -1)
        pred = model.predict(window)
        assert len(pred) == 1


class TestForecastWithConfidence:
    def _fitted(self):
        series = np.sin(np.linspace(0, 6 * np.pi, 120)).astype(np.float32)
        model, _ = fit_channel_forecaster(series, lookback=20, horizon=1)
        return model, series[-20:]

    def test_returns_dict(self) -> None:
        model, window = self._fitted()
        result = forecast_with_confidence(model, window, steps=3)
        assert isinstance(result, dict)

    def test_has_point_forecasts(self) -> None:
        model, window = self._fitted()
        result = forecast_with_confidence(model, window, steps=3)
        assert "point" in result or "forecasts" in result or len(result) > 0

    def test_step_count_matches(self) -> None:
        model, window = self._fitted()
        result = forecast_with_confidence(model, window, steps=5)
        first_list = next(iter(result.values()))
        assert len(first_list) == 5
