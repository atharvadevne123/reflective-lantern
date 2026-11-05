"""Lightweight mathematical utility functions for energy analytics."""

from __future__ import annotations

import math

__all__ = [
    "clamp",
    "geometric_mean",
    "harmonic_mean",
    "is_close",
    "linear_interpolate",
    "moving_average",
    "normalize",
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


def geometric_mean(values: list[float]) -> float:
    """Return the geometric mean of *values* (all must be positive).

    Args:
        values: Non-empty list of strictly positive numbers.

    Returns:
        Geometric mean as a float.

    Raises:
        ValueError: If *values* is empty or any value is non-positive.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be strictly positive")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def harmonic_mean(values: list[float]) -> float:
    """Return the harmonic mean of *values* (all must be non-zero).

    Args:
        values: Non-empty list of non-zero numbers.

    Returns:
        Harmonic mean as a float.

    Raises:
        ValueError: If *values* is empty or any value is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v == 0 for v in values):
        raise ValueError("All values must be non-zero for harmonic mean")
    return len(values) / sum(1.0 / v for v in values)


def normalize(values: list[float], low: float = 0.0, high: float = 1.0) -> list[float]:
    """Rescale *values* to the interval [*low*, *high*] (min-max normalization).

    Args:
        values: Non-empty list of numeric values.
        low: Target minimum. Defaults to 0.0.
        high: Target maximum. Defaults to 1.0.

    Returns:
        Rescaled list; all *low* when min == max.

    Raises:
        ValueError: If *values* is empty or *low* >= *high*.
    """
    if not values:
        raise ValueError("values must not be empty")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    v_min, v_max = min(values), max(values)
    if v_min == v_max:
        return [low] * len(values)
    span_in = v_max - v_min
    span_out = high - low
    return [low + (v - v_min) / span_in * span_out for v in values]
