"""Model monitoring: KS-test drift detection and prediction logging."""

from __future__ import annotations

import logging
import statistics
import time
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import AnomalyLog, DriftLog, PredictionLog

logger = logging.getLogger(__name__)

# In-memory reference window (populated after first training)
_reference_window: list[float] = []
REFERENCE_WINDOW_SIZE = 500
DRIFT_P_THRESHOLD = 0.05


def set_reference_window(values: list[float]) -> None:
    """Set the reference distribution for drift detection."""
    global _reference_window
    _reference_window = list(values[-REFERENCE_WINDOW_SIZE:])
    logger.info("Reference window set: %d samples", len(_reference_window))


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS-test between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "reason": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_P_THRESHOLD),
    }


def check_feature_drift(feature_values: dict[str, list[float]], db: Session) -> list[dict[str, Any]]:
    """Check drift per feature and log results."""
    results = []
    for feature_name, current_vals in feature_values.items():
        ref = _reference_window if _reference_window else current_vals
        result = compute_drift(ref, current_vals)
        result["feature"] = feature_name

        entry = DriftLog(
            feature_name=feature_name,
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=int(result["drift_detected"]),
            checked_at=datetime.utcnow(),
        )
        db.add(entry)
        if result["drift_detected"]:
            logger.warning(
                "Drift detected on feature '%s': KS=%.4f p=%.4f",
                feature_name,
                result["ks_statistic"],
                result["p_value"],
            )
        results.append(result)

    db.commit()
    return results


def log_prediction(
    db: Session,
    building_id: str,
    timestamp: datetime,
    predicted_kwh: float,
    latency_ms: float,
    actual_kwh: float | None = None,
    model_version: str = "1.0.0",
) -> None:
    """Persist a prediction record; rolls back on DB error."""
    entry = PredictionLog(
        building_id=building_id,
        timestamp=timestamp,
        predicted_kwh=predicted_kwh,
        actual_kwh=actual_kwh,
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to log prediction for building %s", building_id)


def log_anomaly(
    db: Session,
    building_id: str,
    timestamp: datetime,
    consumption_kwh: float,
    anomaly_score: float,
    is_anomaly: int,
    severity: str,
) -> None:
    """Persist an anomaly detection record; rolls back on DB error."""
    entry = AnomalyLog(
        building_id=building_id,
        timestamp=timestamp,
        consumption_kwh=consumption_kwh,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        severity=severity,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to log anomaly for building %s", building_id)


def get_prediction_stats(db: Session) -> dict[str, Any]:
    """Aggregate prediction statistics for the /metrics endpoint."""
    total = db.query(PredictionLog).count()
    anomalies = db.query(AnomalyLog).filter(AnomalyLog.is_anomaly == 1).count()
    drifts = db.query(DriftLog).filter(DriftLog.drift_detected == 1).count()
    return {
        "total_predictions": total,
        "total_anomalies_flagged": anomalies,
        "total_drift_events": drifts,
        "reference_window_size": len(_reference_window),
    }


def get_recent_predictions(db: Session, limit: int = 200) -> list[PredictionLog]:
    """Return the most recent prediction log entries."""
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()


def get_drift_summary(db: Session) -> list[dict[str, Any]]:
    """Return the latest drift detection result per feature."""
    from sqlalchemy import func

    subq = (
        db.query(DriftLog.feature_name, func.max(DriftLog.checked_at).label("latest"))
        .group_by(DriftLog.feature_name)
        .subquery()
    )
    rows = (
        db.query(DriftLog)
        .join(subq, (DriftLog.feature_name == subq.c.feature_name) & (DriftLog.checked_at == subq.c.latest))
        .all()
    )
    return [
        {
            "feature_name": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
            "checked_at": r.checked_at.isoformat() if r.checked_at else "",
        }
        for r in rows
    ]


def run_drift_check(db: Session, feature_name: str, reference: list[float], current: list[float]) -> Any:
    """Run a KS drift check and persist the result, returning a simple result object."""
    from types import SimpleNamespace

    result = compute_drift(reference, current)
    entry = DriftLog(
        feature_name=feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=int(result["drift_detected"]),
        checked_at=datetime.utcnow(),
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist drift check for feature %s", feature_name)
    return SimpleNamespace(**result)


def reset_reference_window() -> None:
    """Clear the in-memory reference window (useful in tests)."""
    global _reference_window
    _reference_window = []


class LatencyTimer:
    """Context manager to measure request latency in milliseconds."""

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000, 2)


def get_anomaly_stats(db: Session) -> dict[str, Any]:
    """Return aggregated anomaly statistics from the database.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with total_anomalies, critical_count, warning_count, anomaly_rate,
        and mean_anomaly_score.
    """
    all_anomalies = db.query(AnomalyLog).all()
    total = len(all_anomalies)
    if total == 0:
        return {
            "total_anomalies": 0,
            "critical_count": 0,
            "warning_count": 0,
            "anomaly_rate": 0.0,
            "mean_anomaly_score": 0.0,
        }
    flagged = [a for a in all_anomalies if a.is_anomaly == 1]
    critical = [a for a in flagged if a.severity == "critical"]
    warning = [a for a in flagged if a.severity == "warning"]
    scores = [a.anomaly_score for a in all_anomalies]
    mean_score = round(statistics.mean(scores), 4) if scores else 0.0
    return {
        "total_anomalies": total,
        "critical_count": len(critical),
        "warning_count": len(warning),
        "anomaly_rate": round(len(flagged) / total, 4),
        "mean_anomaly_score": mean_score,
    }


def compute_feature_drift_summary(
    feature_values: dict[str, list[float]],
    reference: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute drift for multiple features without persisting to DB.

    Args:
        feature_values: Mapping of feature name to list of current values.
        reference: Shared reference distribution; falls back to _reference_window.

    Returns:
        List of drift result dicts, one per feature, with 'feature', 'ks_statistic',
        'p_value', and 'drift_detected' keys.
    """
    ref = reference if reference is not None else _reference_window
    results = []
    for name, current in feature_values.items():
        result = compute_drift(ref if ref else current, current)
        result["feature"] = name
        results.append(result)
    return results


def summarize_drift_history(drift_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of drift-check results into aggregate statistics.

    Args:
        drift_results: List of dicts, each with at least 'drift_detected',
            'ks_statistic', and 'p_value' keys (e.g. from compute_drift calls).

    Returns:
        Dict with 'total_checks', 'drift_count', 'drift_rate',
        'mean_ks_statistic', and 'min_p_value'.
    """
    if not drift_results:
        return {
            "total_checks": 0,
            "drift_count": 0,
            "drift_rate": 0.0,
            "mean_ks_statistic": 0.0,
            "min_p_value": 1.0,
        }
    total = len(drift_results)
    drifted = [r for r in drift_results if r.get("drift_detected")]
    ks_values = [float(r.get("ks_statistic", 0.0)) for r in drift_results]
    p_values = [float(r.get("p_value", 1.0)) for r in drift_results]
    return {
        "total_checks": total,
        "drift_count": len(drifted),
        "drift_rate": round(len(drifted) / total, 4),
        "mean_ks_statistic": round(sum(ks_values) / total, 4),
        "min_p_value": round(min(p_values), 4),
    }


def get_reference_window_size() -> int:
    """Return the current number of samples in the in-memory reference window."""
    return len(_reference_window)


def is_reference_window_ready(min_samples: int = 10) -> bool:
    """Return True if the reference window has at least *min_samples* readings.

    Args:
        min_samples: Minimum number of samples required (default 10).

    Returns:
        True if the window is large enough for drift detection.
    """
    return len(_reference_window) >= min_samples


def reference_window_stats() -> dict[str, Any]:
    """Return descriptive statistics for the in-memory reference window."""
    vals = _reference_window
    if not vals:
        return {"size": 0, "mean": None, "min": None, "max": None, "std": None}
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    return {
        "size": n,
        "mean": round(mean, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "std": round(variance**0.5, 4),
    }


__all__ = [
    "LatencyTimer",
    "check_feature_drift",
    "compute_drift",
    "compute_feature_drift_summary",
    "get_anomaly_stats",
    "get_drift_summary",
    "get_prediction_stats",
    "get_recent_predictions",
    "get_reference_window_size",
    "is_reference_window_ready",
    "log_anomaly",
    "log_prediction",
    "reference_window_stats",
    "reset_reference_window",
    "run_drift_check",
    "set_reference_window",
    "summarize_drift_history",
]


def zscore_alert(values: list[float], threshold: float = 3.0) -> list[int]:
    """Return indices of values that exceed the z-score threshold.

    Args:
        values: Numeric series.
        threshold: Number of standard deviations to flag. Default 3.0.

    Returns:
        Sorted list of indices where |z-score| > threshold.

    Raises:
        ValueError: If threshold <= 0 or values has fewer than 2 elements.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if len(values) < 2:
        raise ValueError("Need at least 2 values to compute z-scores")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5
    if std == 0:
        return []
    return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]


def drift_severity(p_value: float, alpha: float = 0.05) -> str:
    """Classify drift severity based on a statistical test p-value.

    Buckets: p >= 0.05 → "low", p >= 0.02 → "medium", p >= 0.005 → "high",
    otherwise "critical".

    Args:
        p_value: p-value from a drift test (KS, chi-squared, etc.).
        alpha: Legacy significance level, retained for backwards compatibility.

    Returns:
        One of "low", "medium", "high", or "critical".
    """
    del alpha  # retained for API compatibility; thresholds are absolute
    if p_value >= 0.05:
        return "low"
    if p_value >= 0.02:
        return "medium"
    if p_value >= 0.005:
        return "high"
    return "critical"


def rolling_anomaly_rate(flags: list[bool], window: int = 10) -> list[float]:
    """Compute rolling anomaly rate using strict valid-window convolution.

    Args:
        flags: Boolean anomaly indicators.
        window: Rolling window size. Default 10.

    Returns:
        List of anomaly rates of length ``max(0, len(flags) - window + 1)``.

    Raises:
        ValueError: If window < 1.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if not flags or len(flags) < window:
        return []
    result: list[float] = []
    for i in range(len(flags) - window + 1):
        chunk = flags[i : i + window]
        result.append(round(sum(chunk) / window, 4))
    return result


def alert_count_by_level(alerts: list[dict]) -> dict[str, int]:
    """Aggregate alert records by their severity level.

    Args:
        alerts: List of alert dicts, each with a 'level' key.

    Returns:
        Dict mapping level -> count.
    """
    counts: dict[str, int] = {}
    for alert in alerts:
        level = str(alert.get("level", "unknown"))
        counts[level] = counts.get(level, 0) + 1
    return counts


def p_value_to_confidence(p_value: float) -> float:
    """Convert a statistical p-value to a confidence percentage.

    confidence = (1 - p_value) * 100, clamped to [0, 100].

    Args:
        p_value: P-value (clamped to [0, 1] before conversion).

    Returns:
        Confidence as a percentage in [0.0, 100.0].
    """
    clamped = max(0.0, min(1.0, p_value))
    return round((1.0 - clamped) * 100.0, 4)


def alert_suppression_window(
    last_alert_ts: float,
    now_ts: float,
    cooldown_seconds: float = 300.0,
) -> bool:
    """Return True if an alert is within the suppression cooldown window.

    Prevents alert storms by suppressing repeated alerts for the same event
    within *cooldown_seconds* of the last alert.

    Args:
        last_alert_ts: Unix timestamp of the most recent alert.
        now_ts: Current Unix timestamp.
        cooldown_seconds: Suppression window in seconds.

    Returns:
        True if the alert should be suppressed (within cooldown).

    Raises:
        ValueError: If cooldown_seconds is not positive.
    """
    if cooldown_seconds <= 0:
        raise ValueError(f"cooldown_seconds must be positive, got {cooldown_seconds}")
    return (now_ts - last_alert_ts) < cooldown_seconds


def drift_trend(p_values: list[float]) -> str:
    """Classify the overall drift trend from a series of p-values.

    Looks at the last 5 p-values relative to the first 5 to determine if
    drift is worsening, improving, or stable.

    Args:
        p_values: Ordered series of p-values (oldest first).

    Returns:
        ``worsening`` — mean of recent half is lower than early half (more drift).
        ``improving`` — mean of recent half is higher than early half.
        ``stable``    — when fewer than 4 values or negligible change.
    """
    if len(p_values) < 4:
        return "stable"
    mid = len(p_values) // 2
    early_mean = sum(p_values[:mid]) / mid
    recent_mean = sum(p_values[mid:]) / (len(p_values) - mid)
    delta = recent_mean - early_mean
    if delta < -0.05:
        return "worsening"
    if delta > 0.05:
        return "improving"
    return "stable"


def alert_rate(alert_counts: list[int], window: int = 7) -> float:
    """Return the average alert count over the last *window* periods.

    Args:
        alert_counts: Time-ordered list of per-period alert counts.
        window: Number of trailing periods to average.

    Returns:
        Mean alert rate; 0.0 for empty input.
    """
    if not alert_counts:
        return 0.0
    tail = alert_counts[-window:]
    return round(sum(tail) / len(tail), 4)


def error_budget_remaining(
    target_slo: float,
    actual_availability: float,
    period_minutes: int = 10080,
) -> float:
    """Return remaining error budget in minutes for a given SLO.

    Args:
        target_slo: Target availability as a fraction in [0, 1].
        actual_availability: Measured availability fraction in [0, 1].
        period_minutes: Length of the SLO measurement window in minutes.

    Returns:
        Remaining budget in minutes (negative if SLO already breached).
    """
    allowed_downtime = (1 - target_slo) * period_minutes
    actual_downtime = (1 - actual_availability) * period_minutes
    return round(allowed_downtime - actual_downtime, 4)


def degradation_severity(error_rate: float) -> str:
    """Classify a service error rate into a degradation severity level.

    Buckets:
        * error_rate < 0.01 → "healthy"
        * 0.01 <= error_rate < 0.05 → "low"
        * 0.05 <= error_rate < 0.20 → "medium"
        * error_rate >= 0.20 → "high"

    Args:
        error_rate: Fraction of requests that errored in [0, 1].

    Returns:
        Severity label ('healthy', 'low', 'medium', or 'high').
    """
    if error_rate < 0.01:
        return "healthy"
    if error_rate < 0.05:
        return "low"
    if error_rate < 0.20:
        return "medium"
    return "high"


def latency_slo_breached(latencies_ms: list[float], slo_ms: float, threshold_pct: float = 0.05) -> bool:
    """Return True when more than *threshold_pct* of latencies exceed *slo_ms*.

    Args:
        latencies_ms: Observed request latencies in milliseconds.
        slo_ms: Service level objective threshold in milliseconds.
        threshold_pct: Fraction of samples above the SLO that constitutes a breach.

    Returns:
        True when the SLO is breached over the sample.

    Raises:
        ValueError: If *slo_ms* is not positive or *threshold_pct* is out of [0, 1].
    """
    if slo_ms <= 0:
        raise ValueError(f"slo_ms must be positive, got {slo_ms}")
    if not 0 <= threshold_pct <= 1:
        raise ValueError(f"threshold_pct must be in [0, 1], got {threshold_pct}")
    if not latencies_ms:
        return False
    over = sum(1 for latency in latencies_ms if latency > slo_ms)
    return (over / len(latencies_ms)) > threshold_pct
