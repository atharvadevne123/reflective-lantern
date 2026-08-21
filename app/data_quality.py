"""Data quality scoring and validation utilities for energy readings."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_VALID_HOUR_RANGE = (0, 23)
_VALID_MONTH_RANGE = (1, 12)
_VALID_DOW_RANGE = (0, 6)


def _check_hour(record: dict[str, Any], issues: list[str]) -> int:
    """Validate the hour field; return the score penalty."""
    hour = record.get("hour")
    if hour is None:
        issues.append("missing:hour")
        return 20
    if not (_VALID_HOUR_RANGE[0] <= int(hour) <= _VALID_HOUR_RANGE[1]):
        issues.append(f"invalid:hour={hour}")
        return 15
    return 0


def _check_month(record: dict[str, Any], issues: list[str]) -> int:
    """Validate the month field; return the score penalty."""
    month = record.get("month")
    if month is None:
        issues.append("missing:month")
        return 10
    if not (_VALID_MONTH_RANGE[0] <= int(month) <= _VALID_MONTH_RANGE[1]):
        issues.append(f"invalid:month={month}")
        return 10
    return 0


def _check_day_of_week(record: dict[str, Any], issues: list[str]) -> int:
    """Validate the day_of_week field; return the score penalty."""
    dow = record.get("day_of_week")
    if dow is None:
        issues.append("missing:day_of_week")
        return 10
    if not (_VALID_DOW_RANGE[0] <= int(dow) <= _VALID_DOW_RANGE[1]):
        issues.append(f"invalid:day_of_week={dow}")
        return 10
    return 0


def _check_consumption(record: dict[str, Any], issues: list[str]) -> int:
    """Validate the consumption_kwh field; return the score penalty."""
    kwh = record.get("consumption_kwh")
    if kwh is None:
        issues.append("missing:consumption_kwh")
        return 20
    if float(kwh) < 0:
        issues.append(f"negative:consumption_kwh={kwh}")
        return 15
    if float(kwh) > 1_000:
        issues.append(f"extreme:consumption_kwh={kwh}")
        return 5
    return 0


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compute a data-quality score (0-100) for a single energy record.

    Each check contributes points. Missing or out-of-range fields deduct points.
    Returns the original record augmented with ``dq_score`` and ``dq_issues``.
    """
    issues: list[str] = []
    penalty = 0
    penalty += _check_hour(record, issues)
    penalty += _check_month(record, issues)
    penalty += _check_day_of_week(record, issues)
    penalty += _check_consumption(record, issues)

    temp = record.get("temperature_c")
    if temp is not None and not (-60 <= float(temp) <= 60):
        issues.append(f"extreme:temperature_c={temp}")
        penalty += 5

    hum = record.get("humidity_pct")
    if hum is not None and not (0 <= float(hum) <= 100):
        issues.append(f"invalid:humidity_pct={hum}")
        penalty += 5

    return {
        **record,
        "dq_score": max(0, 100 - penalty),
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
    logger.debug(
        "quality_summary: n=%d mean=%.1f n_failing=%d", len(scored), summary["mean_score"], summary["n_failing"]
    )
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
    std = variance**0.5
    if std == 0:
        return []
    return [r for r in records if field in r and abs((float(r[field]) - mean) / std) > z_threshold]


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
    """Compute the fraction of populated (field, record) cells for *required_fields*.

    Averages presence across every required field in every record so that a
    record missing half its fields contributes 0.5 to the score.

    Args:
        records: List of dicts to check.
        required_fields: Field names that must be non-None and non-empty.

    Returns:
        Float in [0.0, 1.0]; 1.0 means every required cell is populated.
    """
    if not records or not required_fields:
        return 0.0
    total_cells = len(records) * len(required_fields)
    filled = sum(1 for r in records for f in required_fields if r.get(f) is not None and r.get(f) != "")
    return round(filled / total_cells, 4)


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


def schema_validate(
    record: dict[str, Any],
    schema: dict[str, type],
) -> list[str]:
    """Validate that *record* fields match their expected Python types.

    Args:
        record: The record dict to validate.
        schema: Mapping of field name to expected Python type (e.g. ``{"kwh": float}``).

    Returns:
        List of validation error strings, empty if all fields conform.
    """
    errors: list[str] = []
    for field, expected_type in schema.items():
        value = record.get(field)
        if value is None:
            errors.append(f"missing:{field}")
        elif not isinstance(value, expected_type):
            errors.append(f"type_error:{field} expected {expected_type.__name__}, got {type(value).__name__}")
    return errors


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with string fields stripped and lower-cased.

    Numeric fields are left untouched. Useful for standardising building IDs,
    region names, and building type labels before insertion.

    Args:
        record: Input record dict.

    Returns:
        New dict with string values stripped of leading/trailing whitespace
        and converted to lower-case.
    """
    out: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, str):
            out[k] = v.strip().lower()
        else:
            out[k] = v
    return out


__all__ = [
    "batch_score",
    "completeness_score",
    "cross_field_consistency",
    "cross_field_validation",
    "detect_data_gaps",
    "detect_duplicates",
    "field_entropy",
    "field_type_consistency",
    "field_value_counts",
    "fill_missing",
    "find_duplicate_rows",
    "flag_outliers",
    "normalize_record",
    "null_rate",
    "outlier_count_iqr",
    "quality_summary",
    "range_violation_count",
    "record_completeness",
    "records_missing_field",
    "schema_conformance_rate",
    "schema_validate",
    "score_record",
    "unique_values",
    "validate_date_range",
    "validate_enum_field",
    "value_range_check",
]


def field_value_counts(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    """Return a frequency table of values for *field* across *records*.

    Args:
        records: List of record dicts.
        field: Field name to count distinct values for.

    Returns:
        Dict mapping each distinct value to its occurrence count, sorted by
        count descending.
    """
    counts: dict[str, int] = {}
    for rec in records:
        val = str(rec.get(field, ""))
        counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def null_rate(records: list[dict[str, Any]], field: str) -> float:
    """Return the fraction of records where *field* is None or missing.

    Args:
        records: List of record dicts.
        field: Field name to check for null values.

    Returns:
        Null rate in [0.0, 1.0]; 0.0 for an empty list.
    """
    if not records:
        return 0.0
    null_count = sum(1 for r in records if r.get(field) is None)
    return round(null_count / len(records), 4)


def duplicate_rate(records: list[dict[str, Any]], key_fields: list[str]) -> float:
    """Compute the fraction of records that are duplicates based on *key_fields*.

    Args:
        records: List of record dicts.
        key_fields: Field names that together form a unique key.

    Returns:
        Fraction in [0, 1] of records that are duplicated (non-first occurrences).
    """
    if not records:
        return 0.0
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for rec in records:
        key = tuple(rec.get(f) for f in key_fields)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return round(duplicates / len(records), 4)


def field_completeness(records: list[dict[str, Any]], required_fields: list[str]) -> dict[str, float]:
    """Compute completeness rate (1 - null_rate) for each required field.

    Args:
        records: List of record dicts.
        required_fields: List of field names to check.

    Returns:
        Dict mapping field name to its completeness rate in [0, 1].
    """
    if not records:
        return dict.fromkeys(required_fields, 0.0)
    result = {}
    for field in required_fields:
        present = sum(1 for r in records if r.get(field) is not None)
        result[field] = round(present / len(records), 4)
    return result


def value_range_check(
    values_or_records: list[float] | list[dict[str, Any]],
    *args: Any,
) -> dict[str, Any]:
    """Check how many numeric values fall inside a closed [min, max] range.

    Two calling conventions are supported:

    * ``value_range_check([1.0, 2.0], min_val, max_val)`` — plain sequence.
    * ``value_range_check(records, field, min_val, max_val)`` — extract
      *field* from each dict record.

    Args:
        values_or_records: A list of floats OR a list of record dicts.
        *args: Either ``(min_val, max_val)`` or ``(field, min_val, max_val)``.

    Returns:
        Dict with ``total_checked``, ``in_range``, ``out_of_range``,
        ``out_of_range_count``, and ``out_of_range_rate``.

    Raises:
        ValueError: If the input sequence is empty or ``min_val > max_val``.
    """
    if len(args) == 2:
        min_val, max_val = args
        field = None
        source: list[Any] = list(values_or_records)
        strict_empty = True
    elif len(args) == 3:
        field, min_val, max_val = args
        source = list(values_or_records)
        strict_empty = False
    else:
        raise ValueError(
            f"value_range_check expects (values, min, max) or (records, field, min, max), got {len(args) + 1} args",
        )
    min_val = float(min_val)
    max_val = float(max_val)
    if min_val > max_val:
        raise ValueError(f"min_val {min_val} must be <= max_val {max_val}")
    if not source and strict_empty:
        raise ValueError("value_range_check requires at least one value")

    total = 0
    in_range = 0
    out_of_range = 0
    for item in source:
        if field is not None:
            if not isinstance(item, dict):
                continue
            val = item.get(field)
        else:
            val = item
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        total += 1
        if min_val <= fval <= max_val:
            in_range += 1
        else:
            out_of_range += 1
    rate = round(out_of_range / total, 4) if total > 0 else 0.0
    return {
        "total_checked": total,
        "in_range": in_range,
        "out_of_range": out_of_range,
        "out_of_range_count": out_of_range,
        "out_of_range_rate": rate,
    }


def data_freshness_score(
    records: list[dict[str, Any]],
    timestamp_field: str,
    max_age_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Compute a freshness score based on how recent the records' timestamps are.

    Args:
        records: List of record dicts.
        timestamp_field: Field name containing ISO 8601 or Unix epoch timestamps.
        max_age_seconds: Age threshold in seconds; records older than this are stale.

    Returns:
        Dict with total_records, fresh_count, stale_count, and freshness_rate.
    """
    now = time.time()
    fresh = 0
    stale = 0
    for rec in records:
        ts = rec.get(timestamp_field)
        if ts is None:
            stale += 1
            continue
        try:
            age = now - float(ts)
            if age <= max_age_seconds:
                fresh += 1
            else:
                stale += 1
        except (TypeError, ValueError):
            stale += 1
    total = fresh + stale
    rate = round(fresh / total, 4) if total > 0 else 0.0
    return {"total_records": total, "fresh_count": fresh, "stale_count": stale, "freshness_rate": rate}


def field_type_consistency(records: list[dict[str, Any]], field: str, expected_type: type) -> float:
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
        if (min_val is not None and fv < min_val) or (max_val is not None and fv > max_val):
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


def cross_field_validation(
    record: dict[str, Any],
    rules: list[tuple[str, str, str]],
) -> list[str]:
    """Validate relationships between pairs of fields in a record.

    Each rule is a 3-tuple: (field_a, operator, field_b) where operator is
    one of '<', '<=', '>', '>=', '==', '!='.

    Args:
        record: A single record dict to validate.
        rules: List of (field_a, operator, field_b) tuples.

    Returns:
        List of violation strings (empty when all rules pass).

    Example:
        cross_field_validation(rec, [("start_hour", "<", "end_hour")])
    """
    _ops = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    violations: list[str] = []
    for field_a, op, field_b in rules:
        va = record.get(field_a)
        vb = record.get(field_b)
        if va is None or vb is None:
            violations.append(f"cross_field: '{field_a}' or '{field_b}' missing")
            continue
        check = _ops.get(op)
        if check is None:
            violations.append(f"cross_field: unknown operator '{op}'")
            continue
        try:
            if not check(float(va), float(vb)):
                violations.append(f"cross_field: {field_a}({va}) {op} {field_b}({vb}) violated")
        except (TypeError, ValueError):
            violations.append(f"cross_field: cannot compare '{field_a}' and '{field_b}' (non-numeric)")
    return violations


def record_completeness(record: dict[str, Any], required_fields: list[str]) -> float:
    """Return the fraction of *required_fields* that are present (non-None) in *record*.

    Args:
        record: A single record dict.
        required_fields: List of field names to check.

    Returns:
        Completeness score in [0, 1]; 1.0 if all required fields are present.
        Returns 1.0 if *required_fields* is empty.
    """
    if not required_fields:
        return 1.0
    present = sum(1 for f in required_fields if record.get(f) is not None)
    return round(present / len(required_fields), 4)


def validate_date_range(record: dict[str, Any], start_field: str, end_field: str) -> list[str]:
    """Validate that a date range in *record* is chronologically valid.

    Both fields must be present and the start value must not exceed the end
    value (as comparable objects — datetime, date, or ISO string).

    Args:
        record: Record dict containing the date fields.
        start_field: Key for the range start.
        end_field: Key for the range end.

    Returns:
        List of error strings; empty when the range is valid.
    """
    errors: list[str] = []
    start = record.get(start_field)
    end = record.get(end_field)
    if start is None:
        errors.append(f"'{start_field}' is required")
    if end is None:
        errors.append(f"'{end_field}' is required")
    if start is not None and end is not None:
        try:
            if start > end:  # type: ignore[operator]
                errors.append(f"'{start_field}' must not be after '{end_field}'")
        except TypeError:
            errors.append(f"Cannot compare '{start_field}' and '{end_field}'")
    return errors


def validate_enum_field(record: dict[str, Any], field: str, allowed: list[object]) -> list[str]:
    """Check that *field* in *record* is one of the *allowed* values.

    Args:
        record: Record dict to validate.
        field: Field name to check.
        allowed: Sequence of permitted values.

    Returns:
        List of error strings; empty when the value is valid or the field
        is absent (presence check is not performed here).
    """
    errors: list[str] = []
    value = record.get(field)
    if value is not None and value not in allowed:
        errors.append(f"'{field}' value {value!r} is not in allowed set {allowed!r}")
    return errors


_ALLOWED_RELATIONS: frozenset[str] = frozenset({"a_lte_b", "a_lt_b", "a_eq_b", "a_gte_b", "a_gt_b"})


def cross_field_consistency(
    records: list[dict[str, Any]],
    field_a: str,
    field_b: str,
    relation: str,
) -> list[int]:
    """Return indices of records that violate a two-field relational rule.

    Args:
        records: List of record dicts to check.
        field_a: First field name.
        field_b: Second field name.
        relation: Rule identifier, one of
            ``a_lte_b``, ``a_lt_b``, ``a_eq_b``, ``a_gte_b``, ``a_gt_b``.

    Returns:
        Sorted list of record indices that violate the rule; records missing
        either field are skipped.

    Raises:
        ValueError: If *relation* is not one of the supported identifiers.
    """
    if relation not in _ALLOWED_RELATIONS:
        raise ValueError(
            f"Unsupported relation {relation!r}; expected one of {sorted(_ALLOWED_RELATIONS)}",
        )
    checks = {
        "a_lte_b": lambda a, b: a <= b,
        "a_lt_b": lambda a, b: a < b,
        "a_eq_b": lambda a, b: a == b,
        "a_gte_b": lambda a, b: a >= b,
        "a_gt_b": lambda a, b: a > b,
    }
    predicate = checks[relation]
    violations: list[int] = []
    for i, rec in enumerate(records):
        a = rec.get(field_a)
        b = rec.get(field_b)
        if a is None or b is None:
            continue
        try:
            if not predicate(a, b):
                violations.append(i)
        except TypeError:
            violations.append(i)
    return violations


def outlier_count_iqr(
    records: list[dict[str, Any]],
    field: str,
    multiplier: float = 1.5,
) -> int:
    """Count records whose *field* value lies outside the Tukey IQR fence.

    Args:
        records: List of record dicts to inspect.
        field: Numeric field name to test.
        multiplier: IQR fence multiplier (default 1.5).

    Returns:
        Number of records considered outliers; 0 if *records* has fewer than
        four numeric values (IQR is undefined on very small samples).

    Raises:
        ValueError: If *multiplier* is not positive.
    """
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    values: list[float] = []
    for rec in records:
        val = rec.get(field)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    if len(values) < 4:
        return 0
    values_sorted = sorted(values)
    n = len(values_sorted)
    q1 = values_sorted[n // 4]
    q3 = values_sorted[(3 * n) // 4]
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return sum(1 for v in values if v < lower or v > upper)


def schema_conformance_rate(
    records: list[dict[str, Any]],
    schema: dict[str, type],
) -> float:
    """Return the fraction of records matching a simple ``{field: type}`` schema.

    A record is conforming when every schema field is present, non-None, and
    an instance of the declared type. An empty input returns ``1.0`` — the
    trivially-satisfied case.

    Args:
        records: List of record dicts to check.
        schema: Mapping from field name to expected Python type.

    Returns:
        Float in [0.0, 1.0].
    """
    if not records:
        return 1.0
    conforming = 0
    for rec in records:
        ok = True
        for field, expected_type in schema.items():
            value = rec.get(field)
            if value is None or not isinstance(value, expected_type):
                ok = False
                break
        if ok:
            conforming += 1
    return round(conforming / len(records), 4)


def find_duplicate_rows(
    records: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    """Return records whose composite key (from *key_fields*) appears more than once.

    Args:
        records: List of record dicts.
        key_fields: Fields whose combined values form the deduplication key.

    Returns:
        List of duplicate records (subsequent occurrences beyond the first).
    """
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[dict[str, Any]] = []
    for rec in records:
        key = tuple(rec.get(f) for f in key_fields)
        if key in seen:
            duplicates.append(rec)
        else:
            seen.add(key)
    return duplicates


def field_entropy(records: list[dict[str, Any]], field: str) -> float:
    """Return the Shannon entropy (base-2, bits) of *field* values across records.

    Args:
        records: List of record dicts.
        field: Field name whose value distribution is measured.

    Returns:
        Entropy in bits (float ≥ 0.0); 0.0 for an empty input or a constant
        distribution.
    """
    if not records:
        return 0.0
    from collections import Counter

    counts = Counter(rec.get(field) for rec in records)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    import math as _math

    entropy = 0.0
    for count in counts.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * _math.log2(p)
    return round(entropy, 4)


def constant_columns(records: list[dict[str, Any]]) -> list[str]:
    """Return the names of fields that have the same value across all records.

    Args:
        records: List of record dicts. Empty list returns [].

    Returns:
        Sorted list of field names whose values never vary.
    """
    if not records:
        return []
    all_keys = set(records[0].keys())
    for rec in records[1:]:
        all_keys &= set(rec.keys())
    result = [k for k in all_keys if len({rec.get(k) for rec in records}) == 1]
    return sorted(result)


def consecutive_missing_count(values: list[float | None]) -> int:
    """Count the longest run of consecutive None values in *values*.

    Args:
        values: A list that may contain None for missing entries.

    Returns:
        Length of the longest consecutive None sequence; 0 if no Nones.
    """
    max_run = current_run = 0
    for v in values:
        if v is None:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def column_cardinality(records: list[dict[str, object]]) -> dict[str, int]:
    """Return the number of distinct non-None values per column.

    Args:
        records: List of flat dicts (rows).

    Returns:
        Dict mapping column name to its cardinality (unique value count).
    """
    if not records:
        return {}
    all_keys: set[str] = set().union(*(r.keys() for r in records))
    return {
        key: len({r.get(key) for r in records if r.get(key) is not None})
        for key in sorted(all_keys)
    }


def outlier_summary(
    values: list[float],
    z_threshold: float = 3.0,
) -> dict[str, object]:
    """Summarise outliers in *values* using z-score thresholding.

    Args:
        values: Numeric series with at least 2 elements.
        z_threshold: Z-score beyond which a value is an outlier.

    Returns:
        Dict with keys: count, fraction, min_outlier, max_outlier, indices.
        Returns zeroed summary when std is zero or fewer than 2 values.
    """
    if len(values) < 2:
        return {"count": 0, "fraction": 0.0, "min_outlier": None, "max_outlier": None, "indices": []}
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    std_v = variance ** 0.5
    if std_v == 0.0:
        return {"count": 0, "fraction": 0.0, "min_outlier": None, "max_outlier": None, "indices": []}
    outliers = [(i, v) for i, v in enumerate(values) if abs(v - mean_v) / std_v > z_threshold]
    if not outliers:
        return {"count": 0, "fraction": 0.0, "min_outlier": None, "max_outlier": None, "indices": []}
    outlier_vals = [v for _, v in outliers]
    return {
        "count": len(outliers),
        "fraction": round(len(outliers) / len(values), 6),
        "min_outlier": min(outlier_vals),
        "max_outlier": max(outlier_vals),
        "indices": [i for i, _ in outliers],
    }
