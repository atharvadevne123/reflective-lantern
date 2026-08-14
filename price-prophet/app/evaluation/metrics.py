"""
Evaluation metrics for Price-Prophet.

All functions operate on plain Python lists and use only the standard
library so they can be tested without scikit-learn installed.
"""

from __future__ import annotations

import math


def _check_lengths(actual: list[float], predicted: list[float]) -> None:
    """Raise ValueError if lists are empty or have different lengths."""
    if not actual or not predicted:
        raise ValueError("actual and predicted must not be empty.")
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}.")


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    actual:
        Ground-truth values.
    predicted:
        Model predictions.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If lists have different lengths or are empty.
    """
    _check_lengths(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=False)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    actual:
        Ground-truth values.
    predicted:
        Model predictions.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If lists have different lengths or are empty.
    """
    _check_lengths(actual, predicted)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False)) / len(actual)
    return math.sqrt(mse)


def mape(actual: list[float], predicted: list[float]) -> float:
    """Mean Absolute Percentage Error.

    Zero-valued actuals are skipped to avoid division-by-zero.

    Parameters
    ----------
    actual:
        Ground-truth values.
    predicted:
        Model predictions.

    Returns
    -------
    float
        MAPE as a percentage (0-100+).  Returns ``0.0`` if all actuals
        are zero.

    Raises
    ------
    ValueError
        If lists have different lengths or are empty.
    """
    _check_lengths(actual, predicted)
    errors = []
    for a, p in zip(actual, predicted, strict=False):
        if a != 0.0:
            errors.append(abs((a - p) / a) * 100.0)
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def r_squared(actual: list[float], predicted: list[float]) -> float:
    """Coefficient of determination (R²).

    R^2 = 1 - SS_res / SS_tot.  Returns ``0.0`` when the total variance
    is zero (constant series).

    Parameters
    ----------
    actual:
        Ground-truth values.
    predicted:
        Model predictions.

    Returns
    -------
    float
        R² score; 1.0 is perfect, 0.0 is baseline (mean predictor),
        negative values indicate worse-than-baseline predictions.

    Raises
    ------
    ValueError
        If lists have different lengths or are empty.
    """
    _check_lengths(actual, predicted)
    mean_actual = sum(actual) / len(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def revenue_uplift(baseline_revenue: float, optimised_revenue: float) -> float:
    """Percentage revenue improvement from baseline to optimised.

    Parameters
    ----------
    baseline_revenue:
        Total revenue achieved with the current (non-optimised) pricing.
    optimised_revenue:
        Total revenue achieved with the optimised pricing.

    Returns
    -------
    float
        Uplift as a percentage; ``0.0`` when *baseline_revenue* is zero.
    """
    if baseline_revenue == 0.0:
        return 0.0
    return (optimised_revenue - baseline_revenue) / baseline_revenue * 100.0
