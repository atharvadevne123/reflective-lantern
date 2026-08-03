"""Multi-step ahead energy consumption forecasting utilities."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def naive_forecast(last_value: float, steps: int) -> list[float]:
    """Return a flat naive forecast: repeat *last_value* for *steps* steps."""
    if steps <= 0:
        return []
    return [round(last_value, 4)] * steps


def drift_forecast(
    values: list[float],
    steps: int,
) -> list[float]:
    """Drift (random walk with drift) forecast using historical mean change.

    Projects the average per-step change forward from the last observed value.
    """
    if not values or steps <= 0:
        return []
    if len(values) == 1:
        return [round(values[-1], 4)] * steps
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    avg_drift = sum(diffs) / len(diffs)
    last = values[-1]
    result = []
    for i in range(1, steps + 1):
        result.append(round(last + avg_drift * i, 4))
    return result


def seasonal_naive_forecast(
    values: list[float],
    steps: int,
    period: int = 24,
) -> list[float]:
    """Seasonal naive forecast: copy values from *period* steps back.

    Useful for hourly data where ``period=24`` repeats yesterday's pattern.
    """
    if not values or steps <= 0:
        return []
    n = len(values)
    result = []
    for i in range(steps):
        idx = (n - period + (i % period)) % n
        result.append(round(values[idx], 4))
    return result


def exponential_smoothing_forecast(
    values: list[float],
    steps: int,
    alpha: float = 0.3,
) -> list[float]:
    """Simple exponential smoothing (SES) forecast for *steps* horizons.

    The SES level from the fitted history is used as the point forecast for
    all *steps* horizons (all equal for simple ES).

    Args:
        values: Historical observations (must be non-empty for non-empty output).
        steps: Number of steps ahead to forecast. Returns empty list if <= 0.
        alpha: Smoothing parameter in (0, 1]. Raises ValueError if out of range.

    Returns:
        List of *steps* floats (all equal to the final smoothed level).

    Raises:
        ValueError: If *alpha* is not in (0, 1].
    """
    if not values or steps <= 0:
        return []
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return [round(level, 4)] * steps


def forecast_summary(forecasts: list[float]) -> dict[str, Any]:
    """Return mean, min, max, and total for a forecast list."""
    if not forecasts:
        return {"mean": 0.0, "min": None, "max": None, "total": 0.0, "steps": 0}
    result = {
        "mean": round(sum(forecasts) / len(forecasts), 4),
        "min": min(forecasts),
        "max": max(forecasts),
        "total": round(sum(forecasts), 4),
        "steps": len(forecasts),
    }
    logger.debug("forecast_summary: steps=%d mean=%.4f", result["steps"], result["mean"])
    return result


def ensemble_forecast(
    values: list[float],
    steps: int,
    alpha: float = 0.3,
    period: int = 24,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> list[float]:
    """Combine naive, drift, and seasonal naive forecasts into a weighted ensemble.

    Args:
        values: Historical observations.
        steps: Number of steps to forecast.
        alpha: Smoothing factor for the exponential smoothing component.
        period: Seasonal period for the seasonal naive component.
        weights: Relative weights for (drift, seasonal, exponential_smoothing).

    Returns:
        Blended forecast as a list of *steps* floats.

    Raises:
        ValueError: If *values* is empty, *steps* <= 0, or weights don't sum to a
            positive value.
    """
    if not values or steps <= 0:
        return []
    w_drift, w_seasonal, w_ses = weights
    total_w = w_drift + w_seasonal + w_ses
    if total_w <= 0:
        raise ValueError("weights must sum to a positive value")

    d = drift_forecast(values, steps)
    s = seasonal_naive_forecast(values, steps, period=period)
    e = exponential_smoothing_forecast(values, steps, alpha=alpha)

    result = []
    for i in range(steps):
        blended = (w_drift * d[i] + w_seasonal * s[i] + w_ses * e[i]) / total_w
        result.append(round(blended, 4))
    return result


def forecast_bias(actual: list[float], predicted: list[float]) -> float:
    """Compute mean forecast bias (predicted - actual), indicating over/under-prediction.

    Args:
        actual: Ground-truth observations.
        predicted: Forecasted values.

    Returns:
        Mean bias; positive means over-prediction, negative means under-prediction.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if not actual or not predicted:
        raise ValueError("actual and predicted must not be empty")
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: {len(actual)} vs {len(predicted)}")
    return round(sum(p - a for p, a in zip(predicted, actual, strict=False)) / len(actual), 6)


__all__ = [
    "drift_forecast",
    "ensemble_forecast",
    "exponential_smoothing_forecast",
    "forecast_bias",
    "forecast_summary",
    "naive_forecast",
    "seasonal_naive_forecast",
]
