"""Model performance and operational metrics helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Compute Mean Absolute Error between actual and predicted values.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        MAE rounded to 6 decimal places.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    return round(sum(abs(a - p) for a, p in zip(actual, predicted, strict=False)) / len(actual), 6)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """Compute Root Mean Squared Error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        RMSE rounded to 6 decimal places.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False)) / len(actual)
    return round(mse ** 0.5, 6)


def mean_absolute_percentage_error(
    actual: list[float],
    predicted: list[float],
) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) as a percentage.

    Args:
        actual: Ground-truth values (non-zero required for MAPE).
        predicted: Model predictions.

    Returns:
        MAPE as a percentage rounded to 4 decimal places.
        Zero-valued actuals are skipped.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    valid = [(a, p) for a, p in zip(actual, predicted, strict=False) if a != 0]
    if not valid:
        return 0.0
    return round(sum(abs((a - p) / a) for a, p in valid) / len(valid) * 100, 4)


def r_squared(actual: list[float], predicted: list[float]) -> float:
    """Compute the coefficient of determination (R²).

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        R² value, rounded to 6 decimal places.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    if ss_tot < 1e-12:
        return 1.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False))
    return round(1.0 - ss_res / ss_tot, 6)


def classification_metrics(
    actual: list[int],
    predicted: list[int],
    positive_label: int = 1,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for binary classification.

    Args:
        actual: Ground-truth class labels.
        predicted: Predicted class labels.
        positive_label: The label considered the positive class. Default 1.

    Returns:
        Dict with ``precision``, ``recall``, and ``f1`` keys (all in [0, 1]).

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if not actual:
        raise ValueError("actual must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    tp = sum(1 for a, p in zip(actual, predicted, strict=False) if a == positive_label and p == positive_label)
    fp = sum(1 for a, p in zip(actual, predicted, strict=False) if a != positive_label and p == positive_label)
    fn = sum(1 for a, p in zip(actual, predicted, strict=False) if a == positive_label and p != positive_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


__all__ = [
    "classification_metrics",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r_squared",
    "root_mean_squared_error",
]
