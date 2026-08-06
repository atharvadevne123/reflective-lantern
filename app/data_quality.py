"""Data quality scoring and validation utilities for energy readings."""

from __future__ import annotations

from typing import Any

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
        return {"mean_score": 0.0, "min_score": 0, "max_score": 0, "n_perfect": 0, "n_failing": 0}
    scores = [r["dq_score"] for r in scored]
    return {
        "mean_score": round(sum(scores) / len(scores), 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "n_perfect": sum(1 for s in scores if s == 100),
        "n_failing": sum(1 for s in scores if s < 60),
    }


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
    "field_type_consistency",
    "fill_missing",
    "flag_outliers",
    "quality_summary",
    "range_violation_count",
    "records_missing_field",
    "score_record",
    "unique_values",
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


def field_type_consistency(
    records: list[dict[str, Any]], field: str, expected_type: type
) -> float:
    """Return the fraction of records where *field* has the expected Python type.

    Args:
        records: List of dicts to inspect.
        field: Key to check in each record.
        expected_type: Python type (e.g. ``float``, ``str``, ``int``).

    Returns:
        Consistency score in [0.0, 1.0]; 1.0 means all present values match.
        Returns 1.0 if no records have the field.
    """
    present = [r for r in records if field in r and r[field] is not None]
    if not present:
        return 1.0
    matches = sum(1 for r in present if isinstance(r[field], expected_type))
    return round(matches / len(present), 4)


def range_violation_count(
    records: list[dict[str, Any]],
    field: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> int:
    """Count how many records have *field* outside the given range.

    Args:
        records: List of dicts to check.
        field: Numeric field name.
        min_val: Optional inclusive lower bound.
        max_val: Optional inclusive upper bound.

    Returns:
        Number of records that violate the bounds.
    """
    count = 0
    for r in records:
        v = r.get(field)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if min_val is not None and fv < min_val:
            count += 1
        elif max_val is not None and fv > max_val:
            count += 1
    return count


def fill_missing(
    records: list[dict[str, Any]],
    field: str,
    fill_value: Any = None,
) -> list[dict[str, Any]]:
    """Return a copy of *records* with missing *field* values filled by *fill_value*.

    Args:
        records: Source list of dicts.
        field: Key to fill when absent or None.
        fill_value: Value to substitute; defaults to None.

    Returns:
        New list of dicts with the field populated where it was missing.
    """
    result = []
    for rec in records:
        row = dict(rec)
        if row.get(field) is None:
            row[field] = fill_value
        result.append(row)
    return result


def unique_values(
    records: list[dict[str, Any]],
    field: str,
) -> list[Any]:
    """Return a sorted list of unique non-None values for *field* across *records*.

    Args:
        records: List of dicts to inspect.
        field: Key to collect values from.

    Returns:
        Sorted list of unique values (excluding None); empty list if field absent.
    """
    seen = set()
    result = []
    for rec in records:
        val = rec.get(field)
        if val is not None and val not in seen:
            seen.add(val)
            result.append(val)
    try:
        return sorted(result)
    except TypeError:
        return result


def records_missing_field(
    records: list[dict[str, Any]],
    field: str,
) -> list[int]:
    """Return the indices of records where *field* is absent or None.

    Args:
        records: List of dicts to inspect.
        field: Key to check for presence.

    Returns:
        Sorted list of integer indices where *field* is missing.
    """
    return [i for i, rec in enumerate(records) if rec.get(field) is None]
