"""Extended forecaster tests — second pass with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pytest

from app.forecaster import (
    CONFIDENCE_LEVEL,
    N_BOOTSTRAP,
    fit_channel_forecaster,
    forecast_with_confidence,
    prepare_supervised_data,
)


def _sine_series(n: int = 200) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.05 * np.random.default_rng(42).standard_normal(n)).astype(np.float32)


def test_prepare_supervised_data_shape() -> None:
    series = np.arange(50, dtype=np.float32)
    X, y = prepare_supervised_data(series, lookback=10, horizon=1)
    assert X.shape[1] == 10
    assert len(X) == len(y)


def test_prepare_supervised_data_too_short_raises() -> None:
    series = np.arange(5, dtype=np.float32)
    with pytest.raises(ValueError):
        prepare_supervised_data(series, lookback=10, horizon=5)


@pytest.mark.parametrize("lookback,horizon", [
    (5, 1),
    (10, 3),
    (20, 5),
])
def test_prepare_supervised_data_various_configs(lookback: int, horizon: int) -> None:
    series = np.arange(100, dtype=np.float32)
    X, y = prepare_supervised_data(series, lookback=lookback, horizon=horizon)
    assert X.shape[1] == lookback
    assert len(X) == len(y)


def test_fit_channel_forecaster_returns_model_and_metrics() -> None:
    series = _sine_series(100)
    model, metrics = fit_channel_forecaster(series, lookback=10, horizon=2)
    assert model is not None
    assert "mse" in metrics
    assert "mae" in metrics
    assert "n_samples" in metrics


def test_fit_channel_forecaster_metrics_non_negative() -> None:
    series = _sine_series(100)
    _, metrics = fit_channel_forecaster(series, lookback=10, horizon=2)
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_forecast_with_confidence_returns_dict() -> None:
    series = _sine_series(100)
    model, _ = fit_channel_forecaster(series, lookback=10, horizon=2)
    last_window = series[-10:]
    result = forecast_with_confidence(model, last_window, steps=3)
    assert isinstance(result, dict)


def test_forecast_with_confidence_has_required_keys() -> None:
    series = _sine_series(100)
    model, _ = fit_channel_forecaster(series, lookback=10, horizon=2)
    last_window = series[-10:]
    result = forecast_with_confidence(model, last_window, steps=3)
    assert "point_forecast" in result
    assert "lower" in result
    assert "upper" in result


def test_forecast_with_confidence_steps_match() -> None:
    series = _sine_series(100)
    model, _ = fit_channel_forecaster(series, lookback=10, horizon=2)
    last_window = series[-10:]
    for steps in [1, 3, 5]:
        result = forecast_with_confidence(model, last_window, steps=steps)
        assert len(result["point_forecast"]) == steps


def test_confidence_intervals_ordered() -> None:
    series = _sine_series(100)
    model, _ = fit_channel_forecaster(series, lookback=10, horizon=2)
    last_window = series[-10:]
    result = forecast_with_confidence(model, last_window, steps=5)
    for lo, hi in zip(result["lower"], result["upper"]):
        assert lo <= hi


def test_n_bootstrap_positive() -> None:
    assert N_BOOTSTRAP > 0


def test_confidence_level_in_unit_interval() -> None:
    assert 0.0 < CONFIDENCE_LEVEL < 1.0
