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
    "confidence_interval",
    "drift_forecast",
    "ensemble_forecast",
    "exponential_smoothing_forecast",
    "forecast_bias",
    "forecast_summary",
    "naive_forecast",
    "seasonal_naive_forecast",
    "stepwise_error_growth",
    "weighted_ensemble_forecast",
]


def confidence_interval(
    forecast: list[float],
    std_error: float,
    z_score: float = 1.96,
) -> list[dict[str, float]]:
    """Return confidence intervals for a forecast sequence.

    Args:
        forecast: Point forecasts.
        std_error: Standard error of the forecast (assumed constant across steps).
        z_score: Z multiplier for the desired confidence level (1.96 = 95%).

    Returns:
        List of dicts with 'point', 'lower', 'upper' for each forecast step.
    """
    margin = z_score * std_error
    return [
        {
            "point": round(f, 4),
            "lower": round(f - margin, 4),
            "upper": round(f + margin, 4),
        }
        for f in forecast
    ]


def stepwise_error_growth(
    forecast: list[float],
    base_error: float,
    growth_factor: float = 1.1,
) -> list[float]:
    """Estimate growing forecast uncertainty that compounds step by step.

    Uncertainty at step *k* = base_error * growth_factor^k.

    Args:
        forecast: Point forecasts (length determines the number of steps).
        base_error: Standard error at the first forecast step.
        growth_factor: Multiplicative error growth per step (default 1.1 = 10% growth).

    Returns:
        List of estimated standard errors, one per forecast step.
    """
    return [
        round(base_error * (growth_factor ** k), 6) for k in range(len(forecast))
    ]


def weighted_ensemble_forecast(
    forecasts: list[list[float]],
    weights: list[float],
) -> list[float]:
    """Blend multiple forecast sequences using per-model weights.

    Args:
        forecasts: List of forecast sequences; all must have the same length.
        weights: Non-negative weights (one per forecast sequence).

    Returns:
        Weighted-average forecast of the same length.

    Raises:
        ValueError: If inputs are empty, have mismatched lengths, or weights sum to zero.
    """
    if not forecasts or not weights:
        raise ValueError("forecasts and weights must not be empty")
    if len(forecasts) != len(weights):
        raise ValueError("forecasts and weights must have the same length")
    n_steps = len(forecasts[0])
    if not all(len(f) == n_steps for f in forecasts):
        raise ValueError("all forecast sequences must have the same length")
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("weights must sum to a positive value")
    result = []
    for i in range(n_steps):
        blended = sum(w * f[i] for w, f in zip(weights, forecasts, strict=False)) / total_w
        result.append(round(blended, 4))
    return result
