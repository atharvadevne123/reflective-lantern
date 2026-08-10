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
    "kwh_stats_by_building",
    "normalize_kwh",
    "partition_records",
    "pivot_by_hour",
    "records_to_csv",
    "records_to_json",
    "records_to_jsonl",
    "sort_records",
    "split_by_day",
    "summarize_export",
    "top_buildings_by_kwh",
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


def top_buildings_by_kwh(records: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    """Return the top *n* buildings ranked by total consumption_kwh.

    Args:
        records: List of row dicts, each with ``building_id`` and ``consumption_kwh``.
        n: How many buildings to return.

    Returns:
        List of dicts with ``building_id`` and ``total_kwh``, descending order.
    """
    totals: dict[str, float] = {}
    for rec in records:
        bid = rec.get("building_id")
        kwh = rec.get("consumption_kwh")
        if bid is None or kwh is None:
            continue
        totals[str(bid)] = totals.get(str(bid), 0.0) + float(kwh)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [{"building_id": bid, "total_kwh": round(kwh, 4)} for bid, kwh in ranked[:n]]


def pivot_by_hour(
    records: list[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    """Pivot records into a dict of building_id -> {hour: total_kwh}.

    Args:
        records: List of row dicts with ``building_id``, ``hour``, and ``consumption_kwh``.

    Returns:
        Nested dict keyed by building_id then hour.
    """
    pivot: dict[str, dict[int, float]] = {}
    for rec in records:
        bid = rec.get("building_id")
        hour = rec.get("hour")
        kwh = rec.get("consumption_kwh")
        if bid is None or hour is None or kwh is None:
            continue
        pivot.setdefault(str(bid), {})
        pivot[str(bid)][int(hour)] = pivot[str(bid)].get(int(hour), 0.0) + float(kwh)
    return pivot


def normalize_kwh(
    records: list[dict[str, Any]], target_min: float = 0.0, target_max: float = 1.0
) -> list[dict[str, Any]]:
    """Return a copy of *records* with ``consumption_kwh`` min-max normalised.

    Args:
        records: Source records; must all have ``consumption_kwh``.
        target_min: Lower bound of output range.
        target_max: Upper bound of output range.

    Returns:
        New list of dicts with ``consumption_kwh`` replaced by the scaled value.
    """
    kwh_vals = [float(r["consumption_kwh"]) for r in records if "consumption_kwh" in r]
    if not kwh_vals:
        return [dict(r) for r in records]
    lo, hi = min(kwh_vals), max(kwh_vals)
    span = hi - lo if hi != lo else 1.0
    out = []
    for rec in records:
        row = dict(rec)
        if "consumption_kwh" in row:
            scaled = (float(row["consumption_kwh"]) - lo) / span
            row["consumption_kwh"] = round(target_min + scaled * (target_max - target_min), 6)
        out.append(row)
    return out


def split_by_day(
    records: list[dict[str, object]],
    date_field: str = "timestamp",
) -> dict[str, list[dict[str, object]]]:
    """Group records into buckets by calendar date extracted from *date_field*.

    Extracts the first 10 characters of the date field string (YYYY-MM-DD).
    Records missing or with malformed date fields go into a '_unknown' bucket.

    Args:
        records: List of energy record dicts.
        date_field: Key that contains an ISO-8601 timestamp string.

    Returns:
        Dict mapping date strings to lists of records.
    """
    buckets: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        ts = rec.get(date_field, "")
        day = str(ts)[:10] if ts else "_unknown"
        if len(day) != 10 or not day[:4].isdigit():
            day = "_unknown"
        buckets.setdefault(day, []).append(rec)
    return buckets


def kwh_stats_by_building(
    records: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    """Return per-building summary statistics for ``consumption_kwh``.

    Args:
        records: List of energy record dicts; each must have 'building_id'
            and 'consumption_kwh' keys.

    Returns:
        Dict mapping building_id to a dict with 'total', 'mean', 'min', 'max',
        and 'count'. Buildings missing either key are skipped.
    """
    aggregates: dict[str, list[float]] = {}
    for rec in records:
        bld = rec.get("building_id")
        kwh = rec.get("consumption_kwh")
        if bld is None or kwh is None:
            continue
        key = str(bld)
        aggregates.setdefault(key, []).append(float(kwh))
    result: dict[str, dict[str, float]] = {}
    for bld, vals in aggregates.items():
        result[bld] = {
            "total": round(sum(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
            "min": min(vals),
            "max": max(vals),
            "count": float(len(vals)),
        }
    return result


def records_to_parquet_bytes(
    records: list[dict[str, Any]],
) -> bytes:
    """Serialize records to Parquet format bytes using pandas.

    Returns a bytes object containing the Parquet-encoded data. Requires
    ``pandas`` and ``pyarrow`` (or ``fastparquet``) to be installed.

    Args:
        records: List of energy record dicts to serialize.

    Returns:
        Raw Parquet file bytes.

    Raises:
        ImportError: If pandas or a Parquet engine is not installed.
        ValueError: If records is empty.
    """
    import io as _io

    import pandas as pd

    if not records:
        raise ValueError("records must not be empty")
    df = pd.DataFrame(records)
    buf = _io.BytesIO()
    df.to_parquet(buf, index=False, engine="auto")
    return buf.getvalue()


def flatten_nested_records(
    records: list[dict[str, Any]],
    prefix: str = "",
    separator: str = "_",
) -> list[dict[str, Any]]:
    """Flatten nested dict records into a single level.

    Args:
        records: Records potentially containing nested dicts.
        prefix: Optional key prefix applied to all flattened keys.
        separator: Character used to join nested key segments.

    Returns:
        List of flat dicts suitable for CSV or tabular export.
    """

    def _flatten(d: dict[str, Any], parent: str = "") -> dict[str, Any]:
        items: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}{separator}{k}" if parent else k
            if prefix:
                key = f"{prefix}{separator}{key}" if parent else f"{prefix}{separator}{k}"
            if isinstance(v, dict):
                items.update(_flatten(v, parent=key))
            else:
                items[key] = v
        return items

    return [_flatten(r) for r in records]


def records_missing_fields(records: list[dict[str, Any]], required: list[str]) -> list[int]:
    """Return indices of records that are missing at least one required field.

    Args:
        records: List of record dicts.
        required: Field names that every record must have.

    Returns:
        Sorted list of zero-based indices of incomplete records.
    """
    return sorted(
        i for i, r in enumerate(records)
        if any(field not in r or r[field] is None for field in required)
    )


def records_to_lookup(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Build a lookup dict keyed by a record field.

    When duplicate keys exist the last record wins.

    Args:
        records: List of record dicts.
        key: Field name to use as the lookup key.

    Returns:
        Dict mapping key-values to full record dicts.

    Raises:
        KeyError: If *key* is not present in any record.
    """
    if records and key not in records[0]:
        raise KeyError(f"Field {key!r} not found in records")
    return {str(r[key]): r for r in records}


def rename_record_fields(
    records: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Return copies of *records* with fields renamed according to *mapping*.

    Fields not in *mapping* are kept unchanged. Fields in *mapping* that don't
    exist in a record are silently skipped.

    Args:
        records: List of record dicts.
        mapping: Dict from old field name to new field name.

    Returns:
        New list of dicts with renamed keys.
    """
    result = []
    for record in records:
        new_rec = {}
        for k, v in record.items():
            new_rec[mapping.get(k, k)] = v
        result.append(new_rec)
    return result
