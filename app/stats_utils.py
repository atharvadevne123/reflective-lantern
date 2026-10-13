"""Descriptive statistics helpers operating on plain Python sequences."""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "mean",
    "variance",
    "std_dev",
    "median",
    "percentile",
    "z_scores",
]


def mean(values: Sequence[float]) -> float:
    """Return arithmetic mean of *values*.

    Args:
        values: Non-empty sequence of numbers.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def variance(values: Sequence[float], ddof: int = 0) -> float:
    """Return population (ddof=0) or sample (ddof=1) variance.

    Args:
        values: Non-empty sequence of numbers.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).

    Raises:
        ValueError: If the sequence is too short for the requested ddof.
    """
    n = len(values)
    if n <= ddof:
        raise ValueError(f"need more than {ddof} values for ddof={ddof}")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - ddof)


def std_dev(values: Sequence[float], ddof: int = 0) -> float:
    """Return standard deviation (square root of variance).

    Args:
        values: Non-empty sequence of numbers.
        ddof: Delta degrees of freedom forwarded to :func:`variance`.
    """
    return math.sqrt(variance(values, ddof=ddof))


def median(values: Sequence[float]) -> float:
    """Return the median of *values*.

    Args:
        values: Non-empty sequence of numbers (need not be sorted).

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_v[mid - 1] + sorted_v[mid]) / 2
    return float(sorted_v[mid])


def percentile(values: Sequence[float], p: float) -> float:
    """Return the p-th percentile of *values* using nearest-rank method.

    Args:
        values: Non-empty sequence of numbers.
        p: Percentile in [0, 100].

    Raises:
        ValueError: If *values* is empty or *p* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 <= p <= 100):
        raise ValueError("p must be in [0, 100]")
    sorted_v = sorted(values)
    idx = max(0, math.ceil(p / 100 * len(sorted_v)) - 1)
    return float(sorted_v[idx])


def z_scores(values: Sequence[float]) -> list[float]:
    """Return z-scores for each element in *values*.

    Args:
        values: Non-empty sequence of numbers.

    Returns:
        List of z-scores; all zeros if std_dev is 0.
    """
    if not values:
        raise ValueError("values must not be empty")
    m = mean(values)
    sd = std_dev(values)
    if sd == 0:
        return [0.0] * len(values)
    return [(x - m) / sd for x in values]
