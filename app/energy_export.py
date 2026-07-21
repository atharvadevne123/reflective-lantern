"""Utilities for exporting energy data to CSV and JSON formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


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
    return [
        {"hour": h, "mean_kwh": sum(vals) / len(vals), "count": len(vals)}
        for h, vals in sorted(buckets.items())
    ]


def summarize_export(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a basic summary dict for a batch of energy records."""
    kwh_vals = [float(r["consumption_kwh"]) for r in records if "consumption_kwh" in r]
    return {
        "total_records": len(records),
        "records_with_kwh": len(kwh_vals),
        "total_kwh": round(sum(kwh_vals), 4),
        "mean_kwh": round(sum(kwh_vals) / len(kwh_vals), 4) if kwh_vals else 0.0,
        "min_kwh": min(kwh_vals) if kwh_vals else None,
        "max_kwh": max(kwh_vals) if kwh_vals else None,
    }

__all__ = [
    "aggregate_by_hour",
    "filter_records",
    "records_to_csv",
    "records_to_json",
    "summarize_export",
]
