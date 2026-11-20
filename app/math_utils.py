"""Lightweight mathematical utility functions for energy analytics."""

from __future__ import annotations

import math

__all__ = [
    "clamp",
    "linear_interpolate",
    "moving_average",
    "percent_change",
    "round_to_decimals",
    "safe_divide",
]


def clamp(value: float, low: float, high: float) -> float:
    """Return *value* bounded to the closed interval [*low*, *high*].

    Args:
        value: The value to clamp.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        *value* if within bounds, otherwise the nearest bound.

    Raises:
        ValueError: If *low* > *high*.
    """
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* on zero division.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return when *denominator* is zero. Defaults to 0.0.

    Returns:
        The quotient, or *default* when *denominator* is zero.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def percent_change(old: float, new: float) -> float:
    """Return the percentage change from *old* to *new*.

    Args:
        old: The baseline value.
        new: The new value to compare against *old*.

    Returns:
        Percentage change as a float (e.g. 10.0 means +10 %).
        Returns 0.0 when *old* is zero to avoid ZeroDivisionError.
    """
    if old == 0.0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


def round_to_decimals(value: float, decimals: int = 2) -> float:
    """Round *value* to *decimals* decimal places.

    Args:
        value: The float to round.
        decimals: Number of decimal places. Defaults to 2.

    Returns:
        Rounded float value.
    """
    return round(value, decimals)


def linear_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """Return y at *x* via linear interpolation between (x0, y0) and (x1, y1).

    Args:
        x0: x coordinate of the first point.
        y0: y coordinate of the first point.
        x1: x coordinate of the second point.
        y1: y coordinate of the second point.
        x: The x value to interpolate at.

    Returns:
        Interpolated y value. Returns *y0* when *x0* == *x1*.
    """
    if x0 == x1:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def moving_average(values: list[float], window: int) -> list[float]:
    """Return the simple moving average of *values* with the given *window* size.

    Pads the head with the first available average until enough data exists.

    Args:
        values: Time-ordered list of numeric values.
        window: Number of points to average over. Must be >= 1.

    Returns:
        List of the same length as *values* with moving averages.

    Raises:
        ValueError: If *window* < 1 or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    result: list[float] = []
    for i, _ in enumerate(values):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Return True if *a* and *b* are close within the given tolerances.

    Thin wrapper around :func:`math.isclose` with the same signature.

    Args:
        a: First value.
        b: Second value.
        rel_tol: Relative tolerance. Defaults to 1e-9.
        abs_tol: Absolute tolerance. Defaults to 0.0.

    Returns:
        True if the values are considered equal within tolerance.
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
