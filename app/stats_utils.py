"""Statistical utility functions for Watt-Guard."""

from __future__ import annotations

import logging
import math
import statistics

logger = logging.getLogger(__name__)


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
    return round(sum(abs(a - p) for a, p in zip(actual, predicted, strict=False)) / len(actual), 6)


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
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False)) / len(actual)
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
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=False))
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
    return round(100.0 * sum(abs(a - p) / abs(a) for a, p in zip(actual, predicted, strict=False)) / len(actual), 4)


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


__all__ = [
    "coefficient_of_variation",
    "geometric_mean",
    "harmonic_mean",
    "interquartile_range",
    "mape",
    "mean_absolute_error",
    "median_absolute_deviation",
    "mode_count",
    "normalize_series",
    "percentile",
    "percentile_rank",
    "r_squared",
    "rolling_mean",
    "root_mean_squared_error",
    "trimmed_mean",
    "variance",
    "weighted_average",
    "zscore",
]


def geometric_mean(values: list[float]) -> float:
    """Compute the geometric mean of *values*.

    Args:
        values: List of positive numeric values.

    Returns:
        Geometric mean as a float.

    Raises:
        ValueError: If *values* is empty or any value is non-positive.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive for geometric mean")
    log_sum = sum(math.log(v) for v in values)
    return round(math.exp(log_sum / len(values)), 6)


def harmonic_mean(values: list[float]) -> float:
    """Compute the harmonic mean of *values*.

    Useful for averaging rates and ratios (e.g. energy efficiency metrics).

    Args:
        values: List of positive numeric values.

    Returns:
        Harmonic mean as a float.

    Raises:
        ValueError: If *values* is empty or any value is zero or negative.
    """
    if not values:
        raise ValueError("values must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive for harmonic mean")
    return round(len(values) / sum(1.0 / v for v in values), 6)


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Compute the weighted average of *values*.

    Args:
        values: Numeric observations.
        weights: Non-negative weights (must sum to > 0 and match len(values)).

    Returns:
        Weighted average as a float.

    Raises:
        ValueError: If inputs are empty, have mismatched lengths, or weights sum to zero.
    """
    if not values or not weights:
        raise ValueError("values and weights must not be empty")
    if len(values) != len(weights):
        raise ValueError(f"Length mismatch: {len(values)} values vs {len(weights)} weights")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    return round(sum(v * w for v, w in zip(values, weights, strict=False)) / total_weight, 6)


def normalize_series(values: list[float]) -> list[float]:
    """Min-max normalize a series to the [0, 1] range.

    Args:
        values: List of numeric observations.

    Returns:
        Normalized series; returns list of zeros if all values are identical.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [round((v - lo) / (hi - lo), 6) for v in values]


def interquartile_range(values: list[float]) -> float:
    """Return the interquartile range (IQR = Q3 - Q1) of a series.

    Args:
        values: List of numeric observations (at least 1 element).

    Returns:
        IQR as a float. Returns 0.0 for a single-element list.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)

    return round(_pct(0.75) - _pct(0.25), 6)


def percentile_rank(values: list[float], target: float) -> float:
    """Return the percentile rank of *target* within *values*.

    Percentile rank = (number of values <= target) / total * 100.

    Args:
        values: Reference distribution (non-empty list).
        target: Value whose rank is to be computed.

    Returns:
        Percentile rank in [0.0, 100.0].

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    count = sum(1 for v in values if v <= target)
    return round(count / len(values) * 100.0, 4)


def zscore(values: list[float], value: float) -> float:
    """Compute the Z-score of *value* relative to the population *values*.

    Z-score = (value - mean) / std. Measures how many standard deviations
    a single observation is from the population mean.

    Args:
        values: Reference population of numeric observations.
        value: The observation to score.

    Returns:
        Z-score as a float, rounded to 6 decimal places.
        Returns 0.0 if the population standard deviation is near zero.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    if std < 1e-9:
        return 0.0
    result = round((value - mean) / std, 6)
    if abs(result) > 3.0:
        logger.debug("zscore: extreme value detected z=%.4f (n=%d)", result, n)
    return result


def coefficient_of_variation(values: list[float]) -> float:
    """Compute the coefficient of variation (CV) as std/mean.

    CV expresses variability relative to the mean, useful for comparing
    dispersion across series with different scales.

    Args:
        values: Non-empty list of numeric observations (mean must be non-zero).

    Returns:
        CV as a percentage (std / mean * 100), rounded to 4 decimal places.
        Returns 0.0 when all values are identical.

    Raises:
        ValueError: If *values* is empty or the mean is zero.
    """
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    mean = sum(values) / n
    if abs(mean) < 1e-12:
        raise ValueError(f"Mean is near zero ({mean:.6e}); CV is undefined")
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    if std < 1e-12:
        return 0.0
    return round(std / abs(mean) * 100.0, 4)


def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute Exponential Moving Average (EMA) of a time series.

    EMA weights recent observations more heavily than earlier ones.

    Args:
        values: Time-ordered observations (at least one element).
        alpha: Smoothing factor in (0, 1].  Higher values give more weight to
            recent observations; lower values produce smoother series.

    Returns:
        EMA series of the same length as *values*.

    Raises:
        ValueError: If *values* is empty or *alpha* is outside (0, 1].
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1.0 - alpha) * ema[-1])
    return [round(x, 6) for x in ema]


def winsorize(values: list[float], lower_pct: float = 5.0, upper_pct: float = 95.0) -> list[float]:
    """Clip extreme values to specified percentile bounds (Winsorisation).

    Replaces values below the lower percentile and above the upper percentile
    with the respective boundary values, reducing the influence of outliers.

    Args:
        values: Input list of numeric observations.
        lower_pct: Lower percentile cutoff in [0, 100).
        upper_pct: Upper percentile cutoff in (0, 100].

    Returns:
        Winsorised list of the same length as *values*.

    Raises:
        ValueError: If *values* is empty or percentile bounds are invalid.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 <= lower_pct < upper_pct <= 100):
        raise ValueError(f"Percentiles must satisfy 0 <= lower_pct < upper_pct <= 100, got {lower_pct}, {upper_pct}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lo_idx = int(lower_pct / 100.0 * (n - 1))
    hi_idx = int(upper_pct / 100.0 * (n - 1))
    lo_val = sorted_vals[lo_idx]
    hi_val = sorted_vals[hi_idx]
    return [max(lo_val, min(hi_val, v)) for v in values]


def rolling_std(values: list[float], window: int) -> list[float]:
    """Compute rolling standard deviation over *window* periods.

    Args:
        values: Time-ordered numeric readings.
        window: Number of periods in the rolling window.

    Returns:
        Rolling std list of the same length; leading values use partial windows.

    Raises:
        ValueError: If *window* is less than 2 or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    import statistics

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(round(statistics.pstdev(chunk), 6) if len(chunk) >= 2 else 0.0)
    return result


def compute_entropy(values: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Normalises *values* to a probability distribution before computing entropy.

    Args:
        values: Non-negative numeric counts or weights.

    Returns:
        Shannon entropy in nats (natural logarithm), rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty or all values are zero.
    """
    import math

    if not values:
        raise ValueError("values must not be empty")
    total = sum(values)
    if total <= 0:
        raise ValueError("sum of values must be positive")
    probs = [v / total for v in values if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return round(entropy, 6)


def compute_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two series.

    Args:
        x: First series.
        y: Second series, same length as *x*.

    Returns:
        Pearson r in [-1, 1], rounded to 6 decimal places.

    Raises:
        ValueError: If series are not equal in length or have fewer than 2 elements.
    """
    import math

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("at least 2 observations required")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    denom_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denom_x < 1e-12 or denom_y < 1e-12:
        return 0.0
    return round(num / (denom_x * denom_y), 6)


def compute_skewness(values: list[float]) -> float:
    """Compute the sample skewness of a distribution.

    Uses the Fisher-Pearson standardised moment coefficient (G1).

    Args:
        values: Numeric observations.

    Returns:
        Skewness coefficient, rounded to 6 decimal places.

    Raises:
        ValueError: If fewer than 3 observations are provided.
    """
    import math

    if len(values) < 3:
        raise ValueError("at least 3 observations required for skewness")
    n = len(values)
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 < 1e-12:
        return 0.0
    skew = m3 / (m2**1.5)
    adj = math.sqrt(n * (n - 1)) / (n - 2) * skew
    return round(adj, 6)


def rolling_percentile(values: list[float], window: int, pct: float) -> list[float]:
    """Compute the rolling *pct*-th percentile over *window* periods.

    Args:
        values: Time-ordered numeric readings.
        window: Number of periods in the rolling window.
        pct: Percentile to compute, in [0, 100].

    Returns:
        Rolling percentile list of the same length as *values*.

    Raises:
        ValueError: If *window* < 1, *values* is empty, or *pct* out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = sorted(values[start : i + 1])
        idx = (pct / 100.0) * (len(chunk) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(chunk) - 1)
        val = chunk[lo] + (chunk[hi] - chunk[lo]) * (idx - lo)
        result.append(round(val, 6))
    return result


def running_mean(values: list[float]) -> list[float]:
    """Compute the cumulative running mean at each position.

    Args:
        values: Input numeric series.

    Returns:
        List of running means of the same length as values. Empty if input is empty.
    """
    if not values:
        return []
    result = []
    total = 0.0
    for i, v in enumerate(values, start=1):
        total += v
        result.append(round(total / i, 6))
    return result


def trimmed_mean(values: list[float], trim_pct: float = 0.1) -> float:
    """Compute the trimmed (truncated) mean by removing extreme values.

    Trims *trim_pct* of values from each tail before computing the mean.

    Args:
        values: Non-empty list of numeric values.
        trim_pct: Fraction to remove from each tail (0.0-0.5 exclusive).

    Returns:
        Trimmed mean rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty or *trim_pct* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= trim_pct < 0.5:
        raise ValueError(f"trim_pct must be in [0, 0.5), got {trim_pct}")
    n = len(values)
    k = int(n * trim_pct)
    sorted_vals = sorted(values)
    trimmed = sorted_vals[k : n - k] if k > 0 else sorted_vals
    if not trimmed:
        return round(sum(sorted_vals) / n, 6)
    return round(sum(trimmed) / len(trimmed), 6)


def variance(values: list[float], population: bool = True) -> float:
    """Compute the variance of *values*.

    Args:
        values: Non-empty list of numeric values.
        population: If True compute population variance (÷ n); else sample (÷ n-1).

    Returns:
        Variance rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty (or has < 2 elements for sample variance).
    """
    if not values:
        raise ValueError("values must not be empty")
    if not population and len(values) < 2:
        raise ValueError("Sample variance requires at least 2 values")
    n = len(values)
    mean = sum(values) / n
    sq_diffs = sum((v - mean) ** 2 for v in values)
    divisor = n if population else n - 1
    return round(sq_diffs / divisor, 6)


def mode_count(values: list[float]) -> tuple[float, int]:
    """Return the mode and its frequency from *values*.

    If multiple values share the highest frequency, the smallest is returned.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        Tuple of (mode_value, count).

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    freq: dict[float, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    mode = min(k for k, c in freq.items() if c == max_count)
    return mode, max_count


def median_absolute_deviation(values: list[float]) -> float:
    """Compute the Median Absolute Deviation (MAD) of *values*.

    MAD = median(|xi - median(x)|)

    A robust measure of variability less sensitive to outliers than std dev.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        MAD rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    med = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0 if n % 2 == 0 else float(sorted_vals[mid])
    deviations = sorted(abs(v - med) for v in values)
    n2 = len(deviations)
    mid2 = n2 // 2
    mad = (deviations[mid2 - 1] + deviations[mid2]) / 2.0 if n2 % 2 == 0 else float(deviations[mid2])
    return round(mad, 6)


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute a simple rolling (moving) average over *values*.

    Args:
        values: Time-ordered numeric observations.
        window: Number of periods in the rolling window (must be >= 1).

    Returns:
        List of rolling means (same length as *values*); early elements use a
        growing window (equivalent to min_periods=1).

    Raises:
        ValueError: If *window* < 1 or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    result = []
    for i, _ in enumerate(values):
        lo = max(0, i - window + 1)
        win = values[lo : i + 1]
        result.append(round(sum(win) / len(win), 6))
    return result


def sample_variance(values: list[float]) -> float:
    """Compute the unbiased sample variance (Bessel-corrected) of *values*.

    Uses n-1 in the denominator, making it unbiased for the population variance.
    Prefer this over ``variance(values, population=False)`` when the intended
    semantics of "sample" need to be explicit at the call site.

    Args:
        values: Non-empty list with at least 2 elements.

    Returns:
        Sample variance, rounded to 6 decimal places.

    Raises:
        ValueError: If *values* has fewer than 2 elements.
    """
    if len(values) < 2:
        raise ValueError("sample_variance requires at least 2 values")
    mean = sum(values) / len(values)
    return round(sum((v - mean) ** 2 for v in values) / (len(values) - 1), 6)
