"""Lightweight time-series forecasting: trend + seasonality decomposition."""

from __future__ import annotations

import numpy as np


def linear_trend(values: list[float]) -> dict:
    """Fit a linear trend and return slope, intercept, and next-step forecast."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return {"slope": 0.0, "intercept": float(arr[0]) if n else 0.0, "next": 0.0}
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, arr, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "next": round(intercept + slope * n, 4),
    }


def seasonal_decompose_simple(values: list[float], period: int = 24) -> dict:
    """Remove linear trend and extract mean seasonal component."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < period:
        return {"seasonal": [], "trend": [], "residual": []}

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    trend = slope * x + intercept
    detrended = arr - trend

    full_cycles = n // period
    trimmed = detrended[: full_cycles * period].reshape(full_cycles, period)
    seasonal = trimmed.mean(axis=0)

    residual = detrended - np.tile(seasonal, full_cycles + 1)[:n]
    return {
        "seasonal": seasonal.tolist(),
        "trend": trend.tolist(),
        "residual": residual.tolist(),
    }


def moving_average_forecast(values: list[float], window: int = 24, steps: int = 1) -> list[float]:
    """Forecast `steps` future values using a rolling moving average."""
    arr = list(values)
    for _ in range(steps):
        w = min(window, len(arr))
        arr.append(float(np.mean(arr[-w:])))
    return [round(v, 4) for v in arr[-steps:]]


def forecast_horizon_accuracy(
    actual: list[float],
    predicted: list[float],
    horizon: int = 24,
) -> dict[str, float]:
    """Compute MAE and RMSE for a specific forecast horizon window.

    Args:
        actual: Observed values.
        predicted: Forecasted values; must be the same length as *actual*.
        horizon: Number of steps (from the start) to evaluate.

    Returns:
        Dict with 'mae', 'rmse', and 'n_evaluated'.
    """
    n = min(horizon, len(actual), len(predicted))
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "n_evaluated": 0}
    errors = [actual[i] - predicted[i] for i in range(n)]
    mae = sum(abs(e) for e in errors) / n
    rmse = (sum(e**2 for e in errors) / n) ** 0.5
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "n_evaluated": n}


def exponential_smoothing(
    values: list[float],
    alpha: float = 0.3,
    steps: int = 1,
) -> list[float]:
    """Forecast *steps* future values using simple exponential smoothing.

    Args:
        values: Historical observations (non-empty).
        alpha: Smoothing factor in (0, 1].
        steps: Number of future steps to predict.

    Returns:
        List of *steps* forecasted values.

    Raises:
        ValueError: If *values* is empty or *alpha* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    level = float(values[0])
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return [round(level, 4)] * steps
