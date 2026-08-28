"""
Lightweight runtime metrics accumulator for Price-Prophet.

Records named float observations (e.g. latency, revenue, error rates)
with optional tag metadata.  All data lives in memory; for persistent
metrics use an external system like Prometheus.
"""

from __future__ import annotations

from typing import Any


class MetricsCollector:
    """Append-only store for named float metrics.

    Each observation is stored as a dict with keys ``name``, ``value``,
    and optionally ``tags``.  No timestamps are recorded to keep the
    implementation dependency-free.
    """

    def __init__(self) -> None:
        """Initialise the collector with empty records, counters, and timings."""
        self._records: list[dict[str, Any]] = []
        self._counters: dict[str, float] = {}
        self._timings: dict[str, list[float]] = {}

    def increment(self, name: str, by: float = 1.0) -> None:
        """Increment a named counter by *by* (default 1)."""
        self._counters[name] = self._counters.get(name, 0.0) + by

    def counter(self, name: str) -> float:
        """Return the current value of counter *name*; 0 if unseen."""
        return self._counters.get(name, 0.0)

    def record_timing(self, name: str, value: float) -> None:
        """Append a timing observation *value* for *name*."""
        self._timings.setdefault(name, []).append(value)

    def avg_timing(self, name: str) -> float:
        """Return average timing for *name*; 0.0 if no observations recorded."""
        vals = self._timings.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all counters and average timings."""
        return {
            "counters": dict(self._counters),
            "avg_timings": {k: self.avg_timing(k) for k in self._timings},
        }

    def reset(self) -> None:
        """Clear all counters, timings, and records."""
        self._records.clear()
        self._counters.clear()
        self._timings.clear()

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Append a metric observation.

        Parameters
        ----------
        name:
            Metric name (e.g. ``"request_latency_ms"``).
        value:
            Numeric value of the observation.
        tags:
            Optional dict of key/value metadata (e.g. ``{"model": "linear"}``).
        """
        entry: dict[str, Any] = {"name": name, "value": float(value)}
        if tags is not None:
            entry["tags"] = dict(tags)
        self._records.append(entry)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all recorded metric observations.

        Returns
        -------
        list[dict]
            Shallow copy of the internal records list.
        """
        return list(self._records)

    def get_by_name(self, name: str) -> list[dict[str, Any]]:
        """Return all observations whose ``name`` matches *name*.

        Parameters
        ----------
        name:
            Metric name to filter by.

        Returns
        -------
        list[dict]
        """
        return [r for r in self._records if r["name"] == name]

    def summary(self, name: str) -> dict[str, Any]:
        """Compute basic statistics over all observations for *name*.

        Parameters
        ----------
        name:
            Metric name to summarise.

        Returns
        -------
        dict
            ``{count, mean, min, max}``; all numeric values are
            ``float``.  Returns zeros for an unknown name.
        """
        values = [r["value"] for r in self._records if r["name"] == name]
        count = len(values)
        if count == 0:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": count,
            "mean": sum(values) / count,
            "min": min(values),
            "max": max(values),
        }

    def clear(self) -> None:
        """Remove all recorded observations."""
        self._records.clear()

    def min_timing(self, name: str) -> float:
        """Return minimum timing for *name*; 0.0 if no observations recorded.

        Parameters
        ----------
        name:
            Timing name to query.
        """
        vals = self._timings.get(name, [])
        return min(vals) if vals else 0.0

    def max_timing(self, name: str) -> float:
        """Return maximum timing for *name*; 0.0 if no observations recorded.

        Parameters
        ----------
        name:
            Timing name to query.
        """
        vals = self._timings.get(name, [])
        return max(vals) if vals else 0.0

    def total_count(self) -> int:
        """Return total number of recorded metric observations.

        Returns
        -------
        int
        """
        return len(self._records)

    def merge(self, other: MetricsCollector) -> None:
        """Merge all records, counters, and timings from *other* into self.

        Counters are summed; timing lists are concatenated; records are appended.

        Parameters
        ----------
        other:
            Another MetricsCollector whose data should be absorbed.
        """
        self._records.extend(other._records)
        for name, value in other._counters.items():
            self._counters[name] = self._counters.get(name, 0.0) + value
        for name, vals in other._timings.items():
            self._timings.setdefault(name, []).extend(vals)


# Module-level singleton - import and use this throughout the application.
collector = MetricsCollector()
