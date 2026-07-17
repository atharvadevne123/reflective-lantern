"""Multi-step forecaster with confidence intervals via bootstrap resampling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 50
CONFIDENCE_LEVEL = 0.90


def prepare_supervised_data(
    series: np.ndarray,
    lookback: int = 20,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D time series into supervised (X, y) pairs.

    Each row of X is a window of `lookback` observations and each
    corresponding y is the next value `horizon` steps ahead.
    """
    X_rows, y_rows = [], []
    for i in range(lookback, len(series) - horizon + 1):
        X_rows.append(series[i - lookback : i])
        y_rows.append(series[i + horizon - 1])
    if not X_rows:
        raise ValueError(f"Series too short for lookback={lookback}, horizon={horizon}")
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def fit_channel_forecaster(
    series: np.ndarray,
    lookback: int = 20,
    horizon: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Fit a Random Forest forecaster for a single time-series channel."""
    X, y = prepare_supervised_data(series, lookback=lookback, horizon=horizon)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = float(np.mean((y - y_pred) ** 2))
    mae = float(np.mean(np.abs(y - y_pred)))
    logger.info("Channel forecaster fit: n=%d MAE=%.4f", len(series), mae)
    return model, {"mse": mse, "mae": mae, "n_samples": len(X)}


def forecast_with_confidence(
    model: RandomForestRegressor,
    last_window: np.ndarray,
    steps: int = 5,
) -> dict[str, list[float]]:
    """Produce point forecasts and confidence intervals via tree-level bootstrapping."""
    point_preds: list[float] = []
    tree_preds: list[list[float]] = [[] for _ in range(steps)]

    window = last_window.copy().astype(np.float32)
    for step in range(steps):
        x = window.reshape(1, -1)
        tree_forecasts = np.array([t.predict(x)[0] for t in model.estimators_])
        point = float(np.mean(tree_forecasts))
        point_preds.append(point)

        low_q = (1 - CONFIDENCE_LEVEL) / 2
        high_q = 1 - low_q
        tree_preds[step] = [
            float(np.quantile(tree_forecasts, low_q)),
            float(np.quantile(tree_forecasts, high_q)),
        ]

        window = np.roll(window, -1)
        window[-1] = point

    return {
        "point": point_preds,
        "lower": [t[0] for t in tree_preds],
        "upper": [t[1] for t in tree_preds],
    }


def multi_channel_forecast(
    channel_series: dict[str, np.ndarray],
    lookback: int = 20,
    horizon: int = 5,
) -> dict[str, Any]:
    """Forecast all channels independently and return combined results."""
    results: dict[str, Any] = {}
    for channel, series in channel_series.items():
        try:
            model, metrics = fit_channel_forecaster(series, lookback=lookback, horizon=horizon)
            last_window = (
                series[-lookback:]
                if len(series) >= lookback
                else np.pad(series, (lookback - len(series), 0))
            )
            fc = forecast_with_confidence(model, last_window, steps=horizon)
            results[channel] = {**fc, "metrics": metrics}
        except Exception as e:
            logger.error("Failed to forecast channel '%s': %s", channel, e)
            results[channel] = {"error": str(e)}
    return results
