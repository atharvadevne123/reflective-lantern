"""
Lightweight runtime metrics accumulator for Price-Prophet.

Records named float observations (e.g. latency, revenue, error rates)
with optional tag metadata.  All data lives in memory; for persistent
metrics use an external system like Prometheus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Append-only store for named float metrics.

    Each observation is stored as a dict with keys ``name``, ``value``,
    and optionally ``tags``.  No timestamps are recorded to keep the
    implementation dependency-free.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, Any]] = None,
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
        entry: Dict[str, Any] = {"name": name, "value": float(value)}
        if tags is not None:
            entry["tags"] = dict(tags)
        self._records.append(entry)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all recorded metric observations.

        Returns
        -------
        list[dict]
            Shallow copy of the internal records list.
        """
        return list(self._records)

    def get_by_name(self, name: str) -> List[Dict[str, Any]]:
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

    def summary(self, name: str) -> Dict[str, Any]:
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


# Module-level singleton – import and use this throughout the application.
collector = MetricsCollector()
