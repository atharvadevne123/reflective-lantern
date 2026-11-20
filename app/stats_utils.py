"""Statistical utility functions for Logistics-Flow."""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "mean",
    "median",
    "percentile",
    "std_dev",
    "variance",
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
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    total_w = sum(weights)
    if total_w == 0:
        return values[len(values) // 2]
    paired = sorted(zip(values, weights, strict=False), key=lambda x: x[0])
    cumulative = 0.0
    for val, w in paired:
        cumulative += w
        if cumulative >= total_w / 2.0:
            return float(val)
    return float(paired[-1][0])


def population_variance(values: list[float]) -> float:
    """Compute the population variance (denominator N) of *values*.

    Args:
        values: Numeric sample values.

    Returns:
        Population variance as a float, rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    mean = sum(values) / len(values)
    return round(sum((v - mean) ** 2 for v in values) / len(values), 6)


def sample_std(values: list[float]) -> float:
    """Compute the sample standard deviation (Bessel-corrected, denominator N-1).

    Args:
        values: Numeric sample values.

    Returns:
        Sample standard deviation as a float.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 values for sample standard deviation")
    mean = sum(values) / n
    return round((sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5, 6)


def correlation_coefficient(x: list[float], y: list[float]) -> float:
    """Compute the Pearson correlation coefficient between two series.

    Args:
        x: First numeric series.
        y: Second numeric series (must be same length as *x*).

    Returns:
        Pearson r in [-1, 1]; 0.0 when variance in either series is zero.

    Raises:
        ValueError: If *x* and *y* have different lengths or fewer than 2 points.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 2:
        raise ValueError("Need at least 2 data points")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return 0.0
    return round(cov / denom, 6)


def rolling_rmse(actual: list[float], predicted: list[float], window: int) -> list[float]:
    """Compute rolling RMSE between actual and predicted over a sliding window.

    Args:
        actual: Ground-truth series.
        predicted: Predicted series (same length as actual).
        window: Number of data points per rolling window.

    Returns:
        List of RMSE values; first (window - 1) entries are NaN-padded as 0.0.

    Raises:
        ValueError: If window < 1 or lists have mismatched lengths.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    result: list[float] = []
    for i in range(len(actual)):
        if i < window - 1:
            result.append(0.0)
        else:
            a_w = actual[i - window + 1 : i + 1]
            p_w = predicted[i - window + 1 : i + 1]
            mse = sum((a - p) ** 2 for a, p in zip(a_w, p_w, strict=False)) / window
            result.append(round(mse**0.5, 6))
    return result


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute a parametric confidence interval for the mean of *values*.

    Uses a t-distribution approximation via the standard error.

    Args:
        values: Sample data (must have at least 2 elements).
        confidence: Confidence level in (0, 1).

    Returns:
        (lower, upper) bounds as a tuple of floats.

    Raises:
        ValueError: If values has fewer than 2 elements or confidence is out of range.
    """
    if len(values) < 2:
        raise ValueError("confidence_interval requires at least 2 values")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    import math

    n = len(values)
    mean_v = sum(values) / n
    std_v = (sum((v - mean_v) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std_v / math.sqrt(n)
    # Approximate z-score for common confidence levels
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(round(confidence, 2), 1.96)
    margin = z * se
    return (round(mean_v - margin, 6), round(mean_v + margin, 6))


def signal_to_noise_ratio(values: list[float]) -> float:
    """Compute the signal-to-noise ratio (mean / std) of *values*.

    Args:
        values: Numeric series with at least 2 elements.

    Returns:
        SNR as a float; returns 0.0 when std is zero.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) < 2:
        return 0.0
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
    std_v = variance**0.5
    if std_v == 0.0:
        return 0.0
    return round(abs(mean_v) / std_v, 6)


def outlier_fraction(values: list[float], z_threshold: float = 3.0) -> float:
    """Return the fraction of *values* that are beyond *z_threshold* standard deviations.

    Args:
        values: Numeric series with at least 2 elements.
        z_threshold: Number of standard deviations defining an outlier.

    Returns:
        Fraction in [0.0, 1.0]; 0.0 if std is zero.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) < 2:
        return 0.0
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    std_v = variance**0.5
    if std_v == 0.0:
        return 0.0
    count = sum(1 for v in values if abs(v - mean_v) / std_v > z_threshold)
    return round(count / len(values), 6)


def sign_changes(values: list[float]) -> int:
    """Count the number of times consecutive values change sign.

    Args:
        values: List of numeric values.

    Returns:
        Integer count of zero-crossings (sign changes) in *values*.
    """
    if len(values) < 2:
        return 0
    count = 0
    for i in range(1, len(values)):
        if values[i - 1] * values[i] < 0:
            count += 1
    return count
