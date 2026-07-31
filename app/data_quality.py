"""Data quality scoring and validation utilities for energy readings."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_VALID_HOUR_RANGE = (0, 23)
_VALID_MONTH_RANGE = (1, 12)
_VALID_DOW_RANGE = (0, 6)


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compute a data-quality score (0-100) for a single energy record.

    Each check contributes points. Missing or out-of-range fields deduct points.
    Returns the original record augmented with ``dq_score`` and ``dq_issues``.
    """
    issues: list[str] = []
    score = 100

    hour = record.get("hour")
    if hour is None:
        issues.append("missing:hour")
        score -= 20
    elif not (_VALID_HOUR_RANGE[0] <= int(hour) <= _VALID_HOUR_RANGE[1]):
        issues.append(f"invalid:hour={hour}")
        score -= 15

    month = record.get("month")
    if month is None:
        issues.append("missing:month")
        score -= 10
    elif not (_VALID_MONTH_RANGE[0] <= int(month) <= _VALID_MONTH_RANGE[1]):
        issues.append(f"invalid:month={month}")
        score -= 10

    dow = record.get("day_of_week")
    if dow is None:
        issues.append("missing:day_of_week")
        score -= 10
    elif not (_VALID_DOW_RANGE[0] <= int(dow) <= _VALID_DOW_RANGE[1]):
        issues.append(f"invalid:day_of_week={dow}")
        score -= 10

    kwh = record.get("consumption_kwh")
    if kwh is None:
        issues.append("missing:consumption_kwh")
        score -= 20
    elif float(kwh) < 0:
        issues.append(f"negative:consumption_kwh={kwh}")
        score -= 15
    elif float(kwh) > 1_000:
        issues.append(f"extreme:consumption_kwh={kwh}")
        score -= 5

    temp = record.get("temperature_c")
    if temp is not None and not (-60 <= float(temp) <= 60):
        issues.append(f"extreme:temperature_c={temp}")
        score -= 5

    hum = record.get("humidity_pct")
    if hum is not None and not (0 <= float(hum) <= 100):
        issues.append(f"invalid:humidity_pct={hum}")
        score -= 5

    return {
        **record,
        "dq_score": max(0, score),
        "dq_issues": issues,
    }


def batch_score(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply :func:`score_record` to each record in *records*."""
    return [score_record(r) for r in records]


def quality_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quality scores across a batch of scored records.

    Args:
        scored: Output of :func:`batch_score`.

    Returns:
        Dict with mean_score, min_score, max_score, n_perfect, n_failing (score < 60).
    """
    if not scored:
        return {"total_records": 0, "mean_score": 0.0, "min_score": 0, "max_score": 0, "n_perfect": 0, "n_failing": 0}
    scores = [r["dq_score"] for r in scored]
    summary = {
        "total_records": len(scored),
        "mean_score": round(sum(scores) / len(scores), 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "n_perfect": sum(1 for s in scores if s == 100),
        "n_failing": sum(1 for s in scores if s < 60),
    }
    logger.debug("quality_summary: n=%d mean=%.1f n_failing=%d", len(scored), summary["mean_score"], summary["n_failing"])
    return summary


def flag_outliers(
    records: list[dict[str, Any]],
    field: str,
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Return records whose *field* value is more than *z_threshold* std devs from the mean."""
    vals = [float(r[field]) for r in records if field in r]
    if len(vals) < 2:
        return []
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = variance ** 0.5
    if std == 0:
        return []
    return [
        r for r in records
        if field in r and abs((float(r[field]) - mean) / std) > z_threshold
    ]

__all__ = [
    "batch_score",
    "completeness_score",
    "detect_data_gaps",
    "detect_duplicates",
    "flag_outliers",
    "quality_summary",
    "score_record",
]


def detect_duplicates(
    records: list[dict[str, Any]],
    key_fields: list[str] | None = None,
) -> list[int]:
    """Return indices of duplicate records based on *key_fields*.

    Two records are considered duplicates if all *key_fields* have identical values.
    When *key_fields* is None, all fields are used as the key.

    Args:
        records: List of dicts representing energy readings.
        key_fields: Fields to use as a composite key. Uses all keys if None.

    Returns:
        Sorted list of integer indices for duplicate records (second and later occurrences).
    """
    seen: set[tuple] = set()
    duplicates: list[int] = []
    for i, record in enumerate(records):
        if key_fields is None:
            key = tuple(sorted(record.items()))
        else:
            key = tuple(record.get(f) for f in key_fields)
        if key in seen:
            duplicates.append(i)
        else:
            seen.add(key)
    return duplicates


def completeness_score(records: list[dict[str, Any]], required_fields: list[str]) -> float:
    """Compute the fraction of records that have all *required_fields* populated.

    Args:
        records: List of dicts to check.
        required_fields: Field names that must be non-None and non-empty.

    Returns:
        Float in [0.0, 1.0]; 1.0 means every record is complete.
    """
    if not records:
        return 0.0
    complete = sum(
        1 for r in records
        if all(r.get(f) is not None and r.get(f) != "" for f in required_fields)
    )
    return round(complete / len(records), 4)


def detect_data_gaps(
    timestamps: list[int],
    expected_interval: int = 3600,
) -> list[tuple[int, int]]:
    """Detect gaps in a sequence of Unix timestamps.

    Returns (start, end) timestamp pairs where the gap between consecutive
    readings exceeds *expected_interval* seconds.

    Args:
        timestamps: Sorted list of Unix timestamps (seconds since epoch).
        expected_interval: Expected gap between readings in seconds (default 3600 = 1 hour).

    Returns:
        List of (gap_start, gap_end) tuples for each detected gap.
    """
    if len(timestamps) < 2:
        return []
    gaps: list[tuple[int, int]] = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > expected_interval:
            gaps.append((timestamps[i - 1], timestamps[i]))
    return gaps
