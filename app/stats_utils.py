"""Statistical utility functions for Logistics-Flow."""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "confidence_interval",
    "correlation_coefficient",
    "mean",
    "median",
    "outlier_fraction",
    "percentile",
    "population_variance",
    "rolling_rmse",
    "sample_std",
    "sign_changes",
    "signal_to_noise_ratio",
    "std_dev",
    "variance",
    "weighted_average",
    "weighted_median",
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

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = variance**0.5
    if std_dev == 0:
        return [0.0] * n
    return [(v - mean) / std_dev for v in values]


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


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the weighted median of *values* with corresponding *weights*.

    Args:
        values: Non-empty sequence of numeric values.
        weights: Non-negative weights of the same length as *values*.

    Returns:
        Weighted median as a float.

    Raises:
        ValueError: If inputs are empty, lengths mismatch, or any weight is negative.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    total_w = sum(weights)
    if total_w == 0:
        return float(values[len(values) // 2])
    paired = sorted(zip(values, weights, strict=False), key=lambda x: x[0])
    cumulative = 0.0
    for val, w in paired:
        cumulative += w
        if cumulative >= total_w / 2.0:
            return float(val)
    return float(paired[-1][0])


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the weighted arithmetic average of *values*.

    Args:
        values: Non-empty sequence of numeric values.
        weights: Non-negative weights of the same length as *values*.

    Returns:
        Weighted average as a float.

    Raises:
        ValueError: If inputs are empty, lengths mismatch, any weight is negative,
            or the total weight is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("Total weight must be greater than zero")
    return sum(v * w for v, w in zip(values, weights, strict=False)) / total_w


def geometric_mean(values: Sequence[float]) -> float:
    """Return the geometric mean of *values* (all must be positive).

    Args:
        values: Non-empty sequence of positive numbers.

    Raises:
        ValueError: If *values* is empty or any value is non-positive.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("all values must be positive for geometric mean")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def harmonic_mean(values: Sequence[float]) -> float:
    """Return the harmonic mean of *values* (all must be non-zero).

    Args:
        values: Non-empty sequence of non-zero numbers.

    Raises:
        ValueError: If *values* is empty or any value is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v == 0 for v in values):
        raise ValueError("all values must be non-zero for harmonic mean")
    return len(values) / sum(1.0 / v for v in values)


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Return the mean absolute error between actual and predicted values.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values (same length as actual).

    Raises:
        ValueError: If lists have different lengths or are empty.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual must not be empty")
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=False)) / len(actual)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """Return the root mean squared error between actual and predicted values.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values (same length as actual).

    Raises:
        ValueError: If lists have different lengths or are empty.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual must not be empty")
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False)) / len(actual)
    return math.sqrt(mse)


def r_squared(actual: list[float], predicted: list[float]) -> float:
    """Return the coefficient of determination (R²) between actual and predicted.

    Args:
        actual: Ground-truth values.
        predicted: Predicted values.

    Returns:
        R² score; 1.0 means perfect fit, can be negative for very poor fits.

    Raises:
        ValueError: If lists have different lengths or are empty.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual must not be empty")
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def mape(actual: list[float], predicted: list[float]) -> float:
    """Return the mean absolute percentage error (MAPE).

    Args:
        actual: Ground-truth values (must be non-zero).
        predicted: Predicted values (same length as actual).

    Returns:
        MAPE as a percentage value (e.g. 5.0 for 5%).

    Raises:
        ValueError: If lists have different lengths, are empty, or any actual value is zero.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual must not be empty")
    if any(a == 0 for a in actual):
        raise ValueError("actual values must be non-zero for MAPE")
    return 100.0 * sum(abs((a - p) / a) for a, p in zip(actual, predicted, strict=False)) / len(actual)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return the coefficient of variation (std / mean) as a percentage.

    Args:
        values: Non-empty sequence of numbers with non-zero mean.

    Raises:
        ValueError: If *values* is empty or mean is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    m = sum(values) / n
    if m == 0:
        raise ValueError("mean is zero; coefficient of variation is undefined")
    s = math.sqrt(sum((x - m) ** 2 for x in values) / n)
    return s / abs(m) * 100.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the range [low, high].

    Args:
        value: Value to clamp.
        low: Lower bound.
        high: Upper bound.

    Raises:
        ValueError: If low > high.
    """
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numerator by denominator, returning *default* if denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Return value when denominator is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def running_mean(values: Sequence[float]) -> list[float]:
    """Return the cumulative (running) mean at each position in *values*.

    Args:
        values: Non-empty sequence of numbers.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    result: list[float] = []
    total = 0.0
    for i, v in enumerate(values, start=1):
        total += v
        result.append(total / i)
    return result


def sample_variance(values: Sequence[float]) -> float:
    """Return the sample variance (Bessel-corrected, denominator N-1).

    Args:
        values: Sequence with at least 2 elements.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    n = len(values)
    if n < 2:
        raise ValueError("sample_variance requires at least 2 values")
    m = sum(values) / n
    return sum((x - m) ** 2 for x in values) / (n - 1)


def interquartile_range(values: Sequence[float]) -> float:
    """Return the interquartile range (IQR = Q3 - Q1) of *values*.

    Args:
        values: Non-empty sequence of numbers.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    return q3 - q1


def kurtosis(values: Sequence[float]) -> float:
    """Return the excess kurtosis of *values* (Fisher's definition, normal = 0).

    Args:
        values: Sequence with at least 2 elements.

    Raises:
        ValueError: If fewer than 2 values or std is zero.
    """
    n = len(values)
    if n < 2:
        raise ValueError("kurtosis requires at least 2 values")
    m = sum(values) / n
    s = math.sqrt(sum((x - m) ** 2 for x in values) / n)
    if s == 0:
        raise ValueError("kurtosis is undefined when std is zero")
    return sum(((x - m) / s) ** 4 for x in values) / n - 3.0


def gini_coefficient(values: Sequence[float]) -> float:
    """Return the Gini coefficient measuring inequality in *values*.

    Args:
        values: Non-empty sequence of non-negative numbers.

    Returns:
        Gini coefficient in [0, 1].

    Raises:
        ValueError: If *values* is empty, any value is negative, or sum is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v < 0 for v in values):
        raise ValueError("all values must be non-negative")
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cum_sum = 0.0
    for i, v in enumerate(sorted_v, start=1):
        cum_sum += v
    # Lorenz-based formula
    weighted = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_v, start=1))
    return weighted / (n * total)


def log_return(prices: Sequence[float]) -> list[float]:
    """Return log returns for a price series: ln(p[t] / p[t-1]).

    Args:
        prices: Sequence of positive prices (at least 2 elements).

    Raises:
        ValueError: If fewer than 2 prices or any price is non-positive.
    """
    if len(prices) < 2:
        raise ValueError("log_return requires at least 2 prices")
    if any(p <= 0 for p in prices):
        raise ValueError("all prices must be positive")
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def trimmed_mean(values: Sequence[float], proportion: float = 0.1) -> float:
    """Return the trimmed mean of *values*, removing top and bottom *proportion* of data.

    Args:
        values: Non-empty sequence of numbers.
        proportion: Fraction to trim from each end, in [0, 0.5).

    Raises:
        ValueError: If *values* is empty or *proportion* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 <= proportion < 0.5):
        raise ValueError("proportion must be in [0, 0.5)")
    n = len(values)
    k = int(n * proportion)
    trimmed = sorted(values)[k : n - k] if k > 0 else sorted(values)
    if not trimmed:
        raise ValueError("trimming removed all values")
    return sum(trimmed) / len(trimmed)


def exponential_moving_average(values: Sequence[float], alpha: float = 0.3) -> list[float]:
    """Return the exponential moving average of *values*.

    Args:
        values: Non-empty sequence of numbers.
        alpha: Smoothing factor in (0, 1].

    Raises:
        ValueError: If *values* is empty or *alpha* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    result = [float(values[0])]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def winsorize(values: Sequence[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    """Replace extreme values by their winsorization bounds.

    Args:
        values: Non-empty sequence of numbers.
        lower: Lower quantile bound (0 to upper).
        upper: Upper quantile bound (lower to 1).

    Raises:
        ValueError: If *values* is empty or bounds are out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 <= lower < upper <= 1):
        raise ValueError("lower and upper must satisfy 0 <= lower < upper <= 1")
    lo = percentile(values, lower * 100)
    hi = percentile(values, upper * 100)
    return [max(lo, min(hi, v)) for v in values]


def rolling_std(values: Sequence[float], window: int) -> list[float]:
    """Return rolling standard deviation over a sliding window.

    Args:
        values: Non-empty sequence of numbers.
        window: Window size (>= 2).

    Returns:
        List of std values; first (window - 1) entries are 0.0 (insufficient data).

    Raises:
        ValueError: If window < 2 or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 2:
        raise ValueError("window must be >= 2")
    result: list[float] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(0.0)
        else:
            w = list(values[i - window + 1 : i + 1])
            m = sum(w) / window
            result.append(math.sqrt(sum((x - m) ** 2 for x in w) / window))
    return result


def compute_entropy(values: Sequence[float]) -> float:
    """Return the Shannon entropy of a probability distribution.

    Args:
        values: Non-empty sequence of non-negative numbers that represent probabilities
                or frequencies (automatically normalised if they don't sum to 1).

    Returns:
        Shannon entropy in nats (base-e log).

    Raises:
        ValueError: If *values* is empty or any value is negative.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v < 0 for v in values):
        raise ValueError("all values must be non-negative")
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


def compute_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Pearson correlation coefficient between two sequences.

    Args:
        x: First numeric sequence.
        y: Second numeric sequence (same length as x).

    Returns:
        Pearson r in [-1, 1]; 0.0 when variance in either series is zero.

    Raises:
        ValueError: If lengths differ or sequences are empty.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not x:
        raise ValueError("sequences must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def compute_skewness(values: Sequence[float]) -> float:
    """Return the skewness of *values* (Fisher-Pearson standardised moment).

    Args:
        values: Sequence with at least 3 elements.

    Raises:
        ValueError: If fewer than 3 values or std is zero.
    """
    n = len(values)
    if n < 3:
        raise ValueError("skewness requires at least 3 values")
    m = sum(values) / n
    s = math.sqrt(sum((x - m) ** 2 for x in values) / n)
    if s == 0:
        raise ValueError("skewness is undefined when std is zero")
    return sum(((x - m) / s) ** 3 for x in values) / n


def rolling_percentile(values: Sequence[float], window: int, p: float) -> list[float]:
    """Return rolling p-th percentile over a sliding window.

    Args:
        values: Non-empty sequence of numbers.
        window: Window size (>= 1).
        p: Percentile in [0, 100].

    Returns:
        List of percentile values; first (window - 1) entries are computed on partial windows.

    Raises:
        ValueError: If *values* is empty, window < 1, or p is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 1:
        raise ValueError("window must be >= 1")
    if not (0 <= p <= 100):
        raise ValueError("p must be in [0, 100]")
    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = list(values[start : i + 1])
        result.append(percentile(w, p))
    return result


def normalize_series(values: Sequence[float]) -> list[float]:
    """Return min-max normalised values in [0, 1].

    Args:
        values: Non-empty sequence of numbers.

    Returns:
        List of values scaled to [0, 1]; all 0.5 if min == max.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def zscore(value: float, mean_v: float, std_v: float) -> float:
    """Return the z-score of a single value given population mean and std.

    Args:
        value: Observed value.
        mean_v: Population mean.
        std_v: Population standard deviation.

    Returns:
        0.0 if std_v is zero, otherwise (value - mean_v) / std_v.
    """
    if std_v == 0:
        return 0.0
    return (value - mean_v) / std_v
