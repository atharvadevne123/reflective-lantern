"""Simple time-series forecasting utilities for Sports-Oracle."""

from __future__ import annotations

import numpy as np


def sma(values: list[float], window: int = 5) -> float:
    """Simple moving average of the last `window` values."""
    if not values:
        return 0.0
    tail = values[-window:]
    return float(np.mean(tail))


def linear_trend(values: list[float]) -> float:
    """Slope of a linear fit over the last N values (positive = rising)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def forecast_form(
    historical_form: list[float],
    n_steps: int = 3,
) -> list[float]:
    """Project team form n_steps into the future using linear extrapolation."""
    if len(historical_form) < 2:
        return [sma(historical_form)] * n_steps
    slope = linear_trend(historical_form)
    last = historical_form[-1]
    return [float(np.clip(last + slope * (i + 1), 0.0, 1.0)) for i in range(n_steps)]
