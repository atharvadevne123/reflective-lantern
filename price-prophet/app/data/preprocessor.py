"""
Data preprocessing utilities for Price-Prophet.

All functions operate on plain Python lists so they remain dependency-free
and easy to test in isolation.  They are designed to be composed into a
preprocessing pipeline before feature engineering.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def normalize(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to the [0, 1] range.

    Parameters
    ----------
    values:
        Input numeric values.

    Returns
    -------
    list[float]
        Normalised values.  Returns a list of 0.5 for constant input
        to avoid division-by-zero.
    """
    if not values:
        return []

    lo = min(values)
    hi = max(values)
    span = hi - lo

    if span == 0.0:
        return [0.5] * len(values)

    return [(v - lo) / span for v in values]


def standardize(values: list[float]) -> list[float]:
    """Z-score standardise a list of floats (mean=0, std=1).

    Parameters
    ----------
    values:
        Input numeric values.

    Returns
    -------
    list[float]
        Standardised values.  Returns a list of zeros when the standard
        deviation is zero.
    """
    if not values:
        return []

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    if std == 0.0:
        return [0.0] * n

    return [(v - mean) / std for v in values]


def encode_category(values: list[str]) -> list[int]:
    """Label-encode a list of category strings to integer indices.

    The mapping is determined by the sorted set of unique values so
    that the encoding is stable regardless of the order items appear.

    Parameters
    ----------
    values:
        Input category strings.

    Returns
    -------
    list[int]
        Integer label for each input value.
    """
    if not values:
        return []

    unique_sorted = sorted(set(values))
    mapping = {label: idx for idx, label in enumerate(unique_sorted)}
    return [mapping[v] for v in values]


def fill_missing(
    values: list[Any | None],
    fill_value: Any = 0.0,
) -> list[Any]:
    """Replace ``None`` entries with *fill_value*.

    Parameters
    ----------
    values:
        Input list potentially containing ``None`` entries.
    fill_value:
        Replacement value for missing entries (default 0.0).

    Returns
    -------
    list
        Copy of *values* with ``None`` replaced by *fill_value*.
    """
    return [fill_value if v is None else v for v in values]


def clip_outliers(
    values: list[float],
    z_threshold: float = 3.0,
) -> list[float]:
    """Clip values that are more than *z_threshold* standard deviations
    from the mean.

    Parameters
    ----------
    values:
        Input numeric values.
    z_threshold:
        Number of standard deviations beyond which a value is clipped
        (default 3.0).

    Returns
    -------
    list[float]
        Values with extreme outliers clipped to the threshold boundaries.
    """
    if not values:
        return []

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    if std == 0.0:
        return list(values)

    lower = mean - z_threshold * std
    upper = mean + z_threshold * std
    clipped = [max(lower, min(upper, v)) for v in values]
    logger.debug(
        "clip_outliers: %d/%d values clipped",
        sum(1 for a, b in zip(values, clipped, strict=False) if a != b),
        n,
    )
    return clipped
