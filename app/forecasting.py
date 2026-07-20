"""Multi-step ahead energy consumption forecasting utilities."""

from __future__ import annotations

from typing import Any


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
    return {
        "mean": round(sum(forecasts) / len(forecasts), 4),
        "min": min(forecasts),
        "max": max(forecasts),
        "total": round(sum(forecasts), 4),
        "steps": len(forecasts),
    }

__all__ = [
    "naive_forecast",
    "drift_forecast",
    "seasonal_naive_forecast",
    "exponential_smoothing_forecast",
    "forecast_summary",
]
