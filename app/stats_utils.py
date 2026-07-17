"""Statistical utility functions for Watt-Guard."""

from __future__ import annotations

import math
import statistics


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute error between actual and predicted series.

    Args:
        actual: Ground truth values.
        predicted: Model predictions.

    Returns:
        MAE as a float.

    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not actual or not predicted:
        raise ValueError("Input lists must not be empty")
    if len(actual) != len(predicted):
        raise ValueError(f"Lists must have the same length, got {len(actual)} vs {len(predicted)}")
    return round(sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual), 6)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """Compute root mean squared error between actual and predicted series.

    Args:
        actual: Ground truth values.
        predicted: Model predictions.

    Returns:
        RMSE as a float.

    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not actual or not predicted:
        raise ValueError("Input lists must not be empty")
    if len(actual) != len(predicted):
        raise ValueError(f"Lists must have the same length, got {len(actual)} vs {len(predicted)}")
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    return round(math.sqrt(mse), 6)


def r_squared(actual: list[float], predicted: list[float]) -> float:
    """Compute coefficient of determination (R²) between actual and predicted series.

    Args:
        actual: Ground truth values.
        predicted: Model predictions.

    Returns:
        R² score; 1.0 is perfect, 0.0 means no better than the mean, negative is worse.

    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not actual or not predicted:
        raise ValueError("Input lists must not be empty")
    if len(actual) != len(predicted):
        raise ValueError(f"Lists must have the same length, got {len(actual)} vs {len(predicted)}")
    mean_actual = statistics.mean(actual)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return round(1.0 - ss_res / ss_tot, 6)


def mape(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute percentage error (MAPE).

    Args:
        actual: Ground truth values (must all be non-zero).
        predicted: Model predictions.

    Returns:
        MAPE as a percentage (e.g. 5.0 means 5%).

    Raises:
        ValueError: If lists are empty, have different lengths, or any actual value is zero.
    """
    if not actual or not predicted:
        raise ValueError("Input lists must not be empty")
    if len(actual) != len(predicted):
        raise ValueError(f"Lists must have the same length, got {len(actual)} vs {len(predicted)}")
    if any(a == 0 for a in actual):
        raise ValueError("MAPE is undefined when any actual value is zero")
    return round(100.0 * sum(abs(a - p) / abs(a) for a, p in zip(actual, predicted)) / len(actual), 4)


def coefficient_of_variation(values: list[float]) -> float:
    """Compute the coefficient of variation (CV = std/mean * 100%).

    Args:
        values: List of numeric values (length >= 2, non-zero mean).

    Returns:
        CV as a percentage, or 0.0 for degenerate inputs.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if abs(mean) < 1e-12:
        return 0.0
    std = statistics.pstdev(values)
    return round(100.0 * std / abs(mean), 4)


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of *values* using linear interpolation.

    Args:
        values: List of numeric values.
        p: Percentile to compute (0 <= p <= 100).

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If *values* is empty or *p* is out of [0, 100].
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 6)
