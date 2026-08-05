"""Utilities for exporting energy data to CSV and JSON formats."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def records_to_csv(records: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Serialize *records* to a CSV string.

    Args:
        records: List of row dicts. All dicts should share the same keys.
        columns: Ordered column names. If None, derived from the first record.

    Returns:
        A UTF-8 CSV string with a header row.
    """
    if not records:
        return ""
    if columns is None:
        columns = list(records[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def records_to_json(records: list[dict[str, Any]], indent: int = 2) -> str:
    """Serialize *records* to a pretty-printed JSON string."""
    return json.dumps(records, indent=indent, default=str)


def filter_records(
    records: list[dict[str, Any]],
    min_kwh: float | None = None,
    max_kwh: float | None = None,
    hour: int | None = None,
    building_id: str | None = None,
) -> list[dict[str, Any]]:
    """Filter *records* by optional field constraints.

    Only records with a ``consumption_kwh`` key are filtered by kWh bounds.
    """
    out = []
    for rec in records:
        kwh = rec.get("consumption_kwh")
        if min_kwh is not None and kwh is not None and kwh < min_kwh:
            continue
        if max_kwh is not None and kwh is not None and kwh > max_kwh:
            continue
        if hour is not None and rec.get("hour") != hour:
            continue
        if building_id is not None and rec.get("building_id") != building_id:
            continue
        out.append(rec)
    return out


def aggregate_by_hour(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return mean consumption_kwh grouped by hour (0-23).

    Records without ``hour`` or ``consumption_kwh`` are skipped.
    """
    buckets: dict[int, list[float]] = {}
    for rec in records:
        h = rec.get("hour")
        kwh = rec.get("consumption_kwh")
        if h is None or kwh is None:
            continue
        buckets.setdefault(int(h), []).append(float(kwh))
    return [{"hour": h, "mean_kwh": sum(vals) / len(vals), "count": len(vals)} for h, vals in sorted(buckets.items())]


def summarize_export(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a basic summary dict for a batch of energy records."""
    kwh_vals = [float(r["consumption_kwh"]) for r in records if "consumption_kwh" in r]
    logger.debug("summarize_export: %d records, %d with kwh values", len(records), len(kwh_vals))
    return {
        "total_records": len(records),
        "records_with_kwh": len(kwh_vals),
        "total_kwh": round(sum(kwh_vals), 4),
        "mean_kwh": round(sum(kwh_vals) / len(kwh_vals), 4) if kwh_vals else 0.0,
        "min_kwh": min(kwh_vals) if kwh_vals else None,
        "max_kwh": max(kwh_vals) if kwh_vals else None,
    }


def records_to_jsonl(records: list[dict[str, Any]]) -> str:
    """Serialize *records* to newline-delimited JSON (JSONL) format.

    Each record becomes one JSON line. Useful for streaming large exports.

    Args:
        records: List of row dicts to serialise.

    Returns:
        A string where each line is a valid JSON object, terminated by a newline.
    """
    lines = [json.dumps(rec, default=str) for rec in records]
    return "\n".join(lines) + ("\n" if lines else "")


def deduplicate_records(
    records: list[dict[str, Any]],
    key_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Remove duplicate records based on *key_fields*.

    Preserves the first occurrence of each unique key combination.

    Args:
        records: Input list of record dicts.
        key_fields: Field names used to compute the uniqueness key.
            Defaults to ``["building_id", "timestamp"]``.

    Returns:
        Deduplicated list in original order.
    """
    if key_fields is None:
        key_fields = ["building_id", "timestamp"]
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        key = tuple(rec.get(f) for f in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(rec)
    logger.debug("deduplicate_records: %d -> %d records", len(records), len(out))
    return out


__all__ = [
    "aggregate_by_hour",
    "deduplicate_records",
    "filter_records",
    "partition_records",
    "records_to_csv",
    "records_to_json",
    "records_to_jsonl",
    "sort_records",
    "summarize_export",
]


def sort_records(
    records: list[dict[str, Any]],
    key: str = "timestamp",
    reverse: bool = False,
) -> list[dict[str, Any]]:
    """Return *records* sorted by *key* field.

    Args:
        records: List of record dicts to sort.
        key: Field name to sort by (default 'timestamp').
        reverse: Sort descending when True (default False).

    Returns:
        New sorted list (original is not mutated).
    """
    return sorted(records, key=lambda r: r.get(key, ""), reverse=reverse)


def partition_records(
    records: list[dict[str, Any]],
    field: str,
    value: object,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *records* into two lists: those matching *value* and those not.

    Args:
        records: Input list of record dicts.
        field: Field name to compare.
        value: Value to match against.

    Returns:
        Tuple (matches, non_matches).
    """
    matches = [r for r in records if r.get(field) == value]
    non_matches = [r for r in records if r.get(field) != value]
    return matches, non_matches


def count_records_by_field(records: list[dict], field: str) -> dict[str, int]:
    """Count records grouped by a field's value.

    Args:
        records: List of record dicts.
        field: Field name to group by.

    Returns:
        Dict mapping field value -> record count.
    """
    counts: dict[str, int] = {}
    for rec in records:
        key = str(rec.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def records_to_tsv(records: list[dict], columns: list[str] | None = None) -> str:
    """Serialize records to tab-separated values.

    Args:
        records: List of dicts.
        columns: Ordered column names; if None, use sorted keys from first record.

    Returns:
        TSV string with header row.
    """
    if not records:
        return ""
    cols = columns if columns else sorted(records[0].keys())
    lines = ["\t".join(cols)]
    for rec in records:
        lines.append("\t".join(str(rec.get(c, "")) for c in cols))
    return "\n".join(lines)


def merge_records(base: list[dict], override: list[dict], key: str) -> list[dict]:
    """Merge two record lists, with *override* records taking precedence by key.

    Args:
        base: Base list of records.
        override: Override records that replace or add to base.
        key: Field name used as the merge key.

    Returns:
        Merged list with overrides applied.
    """
    merged: dict[str, dict] = {str(r.get(key)): r for r in base}
    for rec in override:
        merged[str(rec.get(key))] = rec
    return list(merged.values())


def sample_records(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Return a reproducible sample of up to n records.

    Args:
        records: Source records.
        n: Number of records to sample.
        seed: Random seed for reproducibility.

    Returns:
        Sampled subset of records.

    Raises:
        ValueError: If n < 0.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    import random

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n]
