"""Extended anomaly analysis: Z-score, IQR, and multi-metric severity."""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER = 1.5
RATIO_LOW_THRESHOLD = 0.4
RATIO_HIGH_THRESHOLD = 2.5
MIN_REFERENCE_SIZE = 4


def zscore_flag(value: float, mean: float, std: float, threshold: float = 3.0) -> bool:
    """Return True if *value* is more than *threshold* standard deviations from *mean*.

    Args:
        value: The observation to test.
        mean: Distribution mean.
        std: Distribution standard deviation.
        threshold: Number of standard deviations to use as the boundary.

    Returns:
        True when the observation is flagged as anomalous.
    """
    if std < 1e-9:
        return False
    return abs(value - mean) / std > threshold


def iqr_flag(value: float, q1: float, q3: float, k: float = 1.5) -> bool:
    """Return True if *value* is outside the Tukey fence defined by *q1*, *q3*, and *k*.

    Args:
        value: The observation to test.
        q1: First quartile of the reference distribution.
        q3: Third quartile of the reference distribution.
        k: IQR multiplier (default 1.5 = standard Tukey fence).

    Returns:
        True when the observation is flagged as anomalous.
    """
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return value < lower or value > upper


def compute_severity(
    value: float,
    reference: list[float],
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
) -> dict[str, object]:
    """Run both Z-score and IQR tests and combine into a severity label.

    Args:
        value: Consumption reading to evaluate.
        reference: Historical reference window.
        z_threshold: Z-score boundary for flagging.
        iqr_k: IQR fence multiplier.

    Returns:
        Dict with keys 'z_flag', 'iqr_flag', 'severity' ('none'|'warning'|'critical').
    """
    arr = np.array(reference, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    z = zscore_flag(value, mean, std, z_threshold)
    iq = iqr_flag(value, q1, q3, iqr_k)

    both = z and iq
    either = z or iq
    severity = "critical" if both else ("warning" if either else "none")

    logger.debug("Anomaly severity=%s z=%s iqr=%s value=%.2f mean=%.2f", severity, z, iq, value, mean)
    return {"z_flag": z, "iqr_flag": iq, "severity": severity}


def batch_compute_severity(
    values: list[float],
    reference: list[float],
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
) -> list[dict[str, object]]:
    """Run severity analysis on a batch of values against a shared reference.

    Args:
        values: List of consumption readings to evaluate.
        reference: Historical reference window (must have at least MIN_REFERENCE_SIZE elements).
        z_threshold: Z-score boundary for flagging.
        iqr_k: IQR fence multiplier.

    Returns:
        List of severity dicts (same order as *values*), each with 'z_flag', 'iqr_flag',
        'severity', and 'value' keys.
    """
    if len(reference) < MIN_REFERENCE_SIZE:
        logger.warning(
            "Reference window too small (%d < %d); returning 'none' for all values",
            len(reference),
            MIN_REFERENCE_SIZE,
        )
        return [{"z_flag": False, "iqr_flag": False, "severity": "none", "value": v} for v in values]
    results = []
    for v in values:
        result = compute_severity(v, reference, z_threshold, iqr_k)
        result["value"] = v
        results.append(result)
    return results


def anomaly_rate(severities: list[dict[str, object]]) -> float:
    """Return the fraction of readings flagged as anomalous (warning or critical).

    Args:
        severities: List of severity dicts as returned by batch_compute_severity.

    Returns:
        Float in [0, 1] representing the anomaly rate.
    """
    if not severities:
        return 0.0
    flagged = sum(1 for s in severities if s.get("severity") != "none")
    return round(flagged / len(severities), 4)


def compute_percentile_bounds(
    reference: list[float],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict[str, float]:
    """Return lower and upper percentile bounds from a reference distribution.

    Args:
        reference: Historical reference window (at least 2 elements).
        lower_pct: Lower percentile (default 1st percentile).
        upper_pct: Upper percentile (default 99th percentile).

    Returns:
        Dict with 'lower', 'upper', 'median', and 'mean' keys.

    Raises:
        ValueError: If *reference* is empty or percentiles are out of [0, 100].
    """
    if not reference:
        raise ValueError("reference must be non-empty")
    if not (0 <= lower_pct < upper_pct <= 100):
        raise ValueError(f"Percentiles must satisfy 0 <= lower_pct < upper_pct <= 100, got {lower_pct}, {upper_pct}")
    arr = np.array(reference, dtype=float)
    return {
        "lower": round(float(np.percentile(arr, lower_pct)), 4),
        "upper": round(float(np.percentile(arr, upper_pct)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(arr.mean()), 4),
    }


def classify_consumption(
    value: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """Classify a consumption reading as 'low', 'normal', or 'high'.

    Args:
        value: Consumption reading (kWh).
        low_threshold: Upper bound for the 'low' classification.
        high_threshold: Lower bound for the 'high' classification.

    Returns:
        'low' if value < low_threshold, 'high' if value >= high_threshold,
        else 'normal'.

    Raises:
        ValueError: If low_threshold >= high_threshold.
    """
    if low_threshold >= high_threshold:
        raise ValueError(f"low_threshold must be less than high_threshold, got {low_threshold} >= {high_threshold}")
    if value < low_threshold:
        return "low"
    if value >= high_threshold:
        return "high"
    return "normal"


__all__ = [
    "anomaly_rate",
    "anomaly_summary",
    "batch_compute_severity",
    "classify_consumption",
    "compute_percentile_bounds",
    "compute_severity",
    "compute_z_score",
    "consecutive_anomaly_runs",
    "ewma_smooth",
    "flag_anomaly_rate",
    "flag_z_score_outliers",
    "iqr_flag",
    "rolling_anomaly_flag",
    "top_anomalies",
    "zscore_flag",
]


def top_anomalies(
    severities: list[dict[str, object]],
    n: int = 10,
    severity_order: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return the *n* most severe entries from a severity result list.

    Args:
        severities: List of dicts with at least a 'severity' key (from batch_compute_severity).
        n: Maximum number of results to return.
        severity_order: Ordered list from most to least severe; defaults to
            ['critical', 'warning', 'none'].

    Returns:
        Sorted slice of the input list, most severe first, up to length *n*.
    """
    order = severity_order or ["critical", "warning", "none"]
    rank = {label: i for i, label in enumerate(order)}
    default_rank = len(order)

    def _key(entry: dict[str, object]) -> int:
        return rank.get(str(entry.get("severity", "")), default_rank)

    return sorted(severities, key=_key)[:n]


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Compute the Z-score of *value* given a distribution's mean and std.

    Args:
        value: The observation to score.
        mean: Distribution mean.
        std: Distribution standard deviation.

    Returns:
        Z-score as a float, or 0.0 when std is zero (degenerate distribution).
    """
    if std == 0.0:
        return 0.0
    return (value - mean) / std


def flag_z_score_outliers(
    values: list[float],
    threshold: float = 3.0,
) -> list[int]:
    """Return indices of values whose absolute Z-score exceeds *threshold*.

    Args:
        values: List of numeric observations.
        threshold: Absolute Z-score cutoff for flagging (default 3.0).

    Returns:
        Sorted list of integer indices that are statistical outliers.

    Raises:
        ValueError: If *values* is empty or *threshold* is non-positive.
    """
    if not values:
        raise ValueError("values must not be empty")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    import statistics

    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return sorted(i for i, v in enumerate(values) if abs(compute_z_score(v, mean, std)) > threshold)


def flag_anomaly_rate(flags: list[bool]) -> float:
    """Return the fraction of True values in *flags* (anomaly rate).

    Args:
        flags: List of bool anomaly labels.

    Returns:
        Rate in [0, 1], or 0.0 for an empty list.
    """
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)


def consecutive_anomaly_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for consecutive True runs in *flags*.

    Args:
        flags: List of bool anomaly labels.

    Returns:
        List of (start, end) index tuples (inclusive) for each True run.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def ewma_smooth(
    values: list[float],
    alpha: float = 0.3,
) -> list[float]:
    """Apply Exponentially Weighted Moving Average smoothing to *values*.

    Reduces noise in real-time sensor readings while preserving trend.

    Args:
        values: Time-ordered sensor readings.
        alpha: Smoothing factor in (0, 1]. Higher values weight recent observations more.

    Returns:
        Smoothed series of the same length as *values*.

    Raises:
        ValueError: If *values* is empty or *alpha* is out of (0, 1].
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return [round(x, 6) for x in result]


def anomaly_density(flags: list[int], window_size: int = 24) -> list[float]:
    """Compute the density of anomalies in a sliding window.

    At each position *i* the density is the fraction of the trailing *window_size*
    observations that are flagged as anomalous.

    Args:
        flags: Binary anomaly flags (1 = anomaly, 0 = normal).
        window_size: Sliding window length.

    Returns:
        Float list of the same length as *flags* with density values in [0, 1].

    Raises:
        ValueError: If *flags* is empty or *window_size* < 1.
    """
    if not flags:
        raise ValueError("flags must not be empty")
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")
    result = []
    for i in range(len(flags)):
        start = max(0, i - window_size + 1)
        chunk = flags[start : i + 1]
        result.append(round(sum(chunk) / len(chunk), 4))
    return result


def anomaly_burst_score(flags: list[int], burst_window: int = 6) -> float:
    """Score the burstiness of anomalies in *flags*.

    Counts the maximum number of anomalies observed in any contiguous
    *burst_window* of the series, normalised by *burst_window*.

    Args:
        flags: Binary anomaly flags (1 = anomaly, 0 = normal).
        burst_window: Size of the evaluation window.

    Returns:
        Burst score in [0, 1]; 1.0 means all window slots are anomalous.

    Raises:
        ValueError: If *flags* is empty or *burst_window* < 1.
    """
    if not flags:
        raise ValueError("flags must not be empty")
    if burst_window < 1:
        raise ValueError(f"burst_window must be at least 1, got {burst_window}")
    max_in_window = 0
    for i in range(len(flags)):
        window = flags[i : i + burst_window]
        max_in_window = max(max_in_window, sum(window))
    return round(max_in_window / burst_window, 4)


def percentile_anomaly_flag(
    value: float,
    reference: list[float],
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> bool:
    """Flag *value* as anomalous if it falls outside the *reference* percentile range.

    Args:
        value: Observation to evaluate.
        reference: Reference distribution of historical values.
        lower_pct: Lower percentile bound in [0, 100).
        upper_pct: Upper percentile bound in (0, 100].

    Returns:
        True when *value* is outside the [lower_pct, upper_pct] range.

    Raises:
        ValueError: If *reference* is empty or percentile bounds are invalid.
    """
    if not reference:
        raise ValueError("reference must not be empty")
    if not (0 <= lower_pct < upper_pct <= 100):
        raise ValueError(f"Percentiles must satisfy 0 <= lower < upper <= 100, got {lower_pct}, {upper_pct}")
    sorted_ref = sorted(reference)
    n = len(sorted_ref)
    lo_idx = int(lower_pct / 100.0 * (n - 1))
    hi_idx = int(upper_pct / 100.0 * (n - 1))
    lo_val = sorted_ref[lo_idx]
    hi_val = sorted_ref[hi_idx]
    return value < lo_val or value > hi_val


def consecutive_normal_runs(flags: list[int]) -> list[tuple[int, int]]:
    """Return spans of consecutive normal (0) observations.

    Args:
        flags: Binary anomaly flags (1 = anomaly, 0 = normal).

    Returns:
        List of (start, end) index pairs for consecutive normal spans.
    """
    if not flags:
        return []
    runs = []
    start = None
    for i, f in enumerate(flags):
        if f == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def anomaly_summary(severities: list[dict[str, object]]) -> dict[str, int]:
    """Return a count of each severity level from a batch severity result list.

    Args:
        severities: List of severity dicts (each must have a 'severity' key).

    Returns:
        Dict with 'none', 'warning', 'critical' counts.
    """
    counts: dict[str, int] = {"none": 0, "warning": 0, "critical": 0}
    for entry in severities:
        key = str(entry.get("severity", "none"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def rolling_anomaly_flag(
    values: list[float],
    window: int = 5,
    threshold: float = 3.0,
) -> list[bool]:
    """Flag each value where the rolling Z-score exceeds *threshold*.

    Uses a centered sliding window of size *window* to compute local mean and
    standard deviation; the first and last ``window // 2`` elements use the
    nearest available window.

    Args:
        values: Time-ordered numeric observations.
        window: Width of the rolling window (must be >= 2).
        threshold: Z-score absolute value above which a point is flagged.

    Returns:
        List of bools (same length as *values*), True where anomalous.

    Raises:
        ValueError: If *values* is empty or *window* < 2.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    n = len(values)
    flags: list[bool] = []
    for i in range(n):
        lo = max(0, i - window)
        win = values[lo:i]
        if len(win) < 2:
            flags.append(False)
            continue
        arr = np.array(win, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std == 0.0:
            flags.append(values[i] != mean)
        else:
            flags.append(abs(compute_z_score(values[i], mean, std)) > threshold)
    return flags


def anomaly_free_streak(flags: list[bool]) -> int:
    """Return the length of the current (trailing) anomaly-free streak.

    Counts backward from the last element until a True (anomalous) flag is
    encountered. Useful for dashboard widgets showing "days without anomaly."

    Args:
        flags: Ordered list of anomaly flags (True = anomalous).

    Returns:
        Number of consecutive False values at the end of the list.
        Returns 0 if the list is empty or ends with True.
    """
    streak = 0
    for flag in reversed(flags):
        if flag:
            break
        streak += 1
    return streak


def anomaly_transition_count(flags: list[bool]) -> int:
    """Count the number of times the anomaly state changes.

    A transition is any adjacent pair (flags[i], flags[i+1]) where the values
    differ.  Useful for measuring how "bursty" vs "sustained" anomalies are.

    Args:
        flags: Ordered list of anomaly flags.

    Returns:
        Number of state transitions (0 if list has fewer than 2 elements).
    """
    if len(flags) < 2:
        return 0
    return sum(1 for a, b in itertools.pairwise(flags) if a != b)


def anomaly_rate_by_hour(
    readings: list[tuple[int, bool]],
) -> dict[int, float]:
    """Compute the anomaly rate for each hour-of-day slot.

    Args:
        readings: List of ``(hour, is_anomaly)`` tuples where *hour* is 0-23
            and *is_anomaly* is True/False.

    Returns:
        Dict mapping hour → fraction of readings in that hour that are anomalies.
        Hours with no readings are omitted.

    Raises:
        ValueError: If any hour is outside 0-23.
    """
    if any(not (0 <= h <= 23) for h, _ in readings):
        raise ValueError("All hour values must be in 0-23")
    bucket_total: dict[int, int] = {}
    bucket_anomaly: dict[int, int] = {}
    for hour, flag in readings:
        bucket_total[hour] = bucket_total.get(hour, 0) + 1
        bucket_anomaly[hour] = bucket_anomaly.get(hour, 0) + (1 if flag else 0)
    return {h: round(bucket_anomaly[h] / bucket_total[h], 4) for h in bucket_total}


def severity_weighted_score(
    anomalies: list[dict],
    severity_weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted severity score from a list of anomaly records.

    Args:
        anomalies: List of dicts, each with a ``"severity"`` key (``"critical"``,
            ``"warning"``, or ``"normal"``).
        severity_weights: Custom weights per severity level. Defaults to
            ``{"critical": 3.0, "warning": 1.5, "normal": 0.0}``.

    Returns:
        Total weighted score, rounded to 4 decimal places.
    """
    if severity_weights is None:
        severity_weights = {"critical": 3.0, "warning": 1.5, "normal": 0.0}
    total = 0.0
    for a in anomalies:
        level = str(a.get("severity", "normal"))
        total += severity_weights.get(level, 0.0)
    return round(total, 4)


def batch_anomaly_flag(
    values: list[float],
    mean: float,
    std: float,
    threshold: float = 3.0,
) -> list[bool]:
    """Flag values that deviate more than *threshold* standard deviations from *mean*.

    Args:
        values: Numeric readings to evaluate.
        mean: Reference mean of the distribution.
        std: Reference standard deviation (must be > 0).
        threshold: Number of standard deviations to flag. Default 3.0.

    Returns:
        Boolean list of the same length as *values*; True indicates an anomaly.

    Raises:
        ValueError: If *std* <= 0 or *threshold* <= 0.
    """
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    return [abs(v - mean) > threshold * std for v in values]


def anomaly_score_ema(
    flags: list[bool],
    alpha: float = 0.3,
) -> list[float]:
    """Compute exponential moving average of anomaly flags.

    Produces a smoothed score in [0, 1] that decays toward zero when
    anomalies cease, useful for building burst-momentum dashboards.

    Args:
        flags: Boolean anomaly indicators (True = anomaly).
        alpha: EMA smoothing factor in (0, 1]. Higher values give more weight
            to recent observations. Default 0.3.

    Returns:
        List of smoothed scores the same length as *flags*.

    Raises:
        ValueError: If *alpha* is outside (0, 1].
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    result: list[float] = []
    ema = 0.0
    for f in flags:
        ema = alpha * float(f) + (1.0 - alpha) * ema
        result.append(round(ema, 6))
    return result


def mean_time_between_anomalies(flags: list[bool]) -> float:
    """Compute the mean number of steps between anomaly events.

    Args:
        flags: Boolean sequence of anomaly indicators.

    Returns:
        Average gap (in steps) between consecutive True values, or
        ``float('inf')`` if fewer than two anomalies are present.
    """
    indices = [i for i, f in enumerate(flags) if f]
    if len(indices) < 2:
        return float("inf")
    gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    return round(sum(gaps) / len(gaps), 4)


def anomaly_peak_ratio(values: list[float], flags: list[bool]) -> float:
    """Ratio of the mean anomalous value to the mean normal value.

    Quantifies how much larger (or smaller) anomalous readings are compared
    to the baseline. Returns 0.0 when either group is empty.

    Args:
        values: Numeric observation series.
        flags: Boolean anomaly indicators of the same length.

    Returns:
        Ratio (anomaly mean / normal mean), rounded to 4 decimal places.

    Raises:
        ValueError: If *values* and *flags* differ in length.
    """
    if len(values) != len(flags):
        raise ValueError("values and flags must be the same length")
    anomalous = [v for v, f in zip(values, flags, strict=False) if f]
    normal = [v for v, f in zip(values, flags, strict=False) if not f]
    if not anomalous or not normal:
        return 0.0
    return round(sum(anomalous) / len(anomalous) / (sum(normal) / len(normal)), 4)
