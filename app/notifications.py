"""Alert and notification helpers for energy anomaly events."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A single alert event with severity, message, and optional metadata."""

    severity: str  # "info" | "warning" | "critical"
    message: str
    source: str = "system"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
        }


_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def severity_rank(severity: str) -> int:
    """Return a numeric rank for *severity* (higher = more severe)."""
    return _SEVERITY_RANK.get(severity.lower(), -1)


class AlertQueue:
    """In-memory queue of recent alerts with a configurable max size."""

    def __init__(self, max_size: int = 200) -> None:
        self._max = max_size
        self._alerts: list[Alert] = []

    def push(self, alert: Alert) -> None:
        """Append *alert* and evict the oldest entry when over capacity."""
        if len(self._alerts) >= self._max:
            self._alerts.pop(0)
        self._alerts.append(alert)
        logger.debug("AlertQueue.push: severity=%s source=%s queue_size=%d", alert.severity, alert.source, len(self._alerts))

    def filter_by_severity(self, min_severity: str) -> list[Alert]:
        """Return alerts at or above *min_severity*."""
        min_rank = severity_rank(min_severity)
        return [a for a in self._alerts if severity_rank(a.severity) >= min_rank]

    def filter_by_tag(self, tag: str) -> list[Alert]:
        """Return alerts that contain *tag*."""
        return [a for a in self._alerts if tag in a.tags]

    def clear(self) -> None:
        """Remove all alerts."""
        self._alerts.clear()

    def summary(self) -> dict[str, int]:
        """Count alerts per severity level."""
        counts: dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
        for a in self._alerts:
            key = a.severity.lower()
            counts[key] = counts.get(key, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._alerts)

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)


def make_anomaly_alert(
    building_id: str,
    score: float,
    is_critical: bool = False,
) -> Alert:
    """Build a pre-filled :class:`Alert` for an anomaly detection event."""
    severity = "critical" if is_critical else "warning"
    return Alert(
        severity=severity,
        message=f"Anomaly detected for building {building_id} (score={score:.3f})",
        source="anomaly-detector",
        tags=["anomaly", building_id],
        metadata={"building_id": building_id, "anomaly_score": score},
    )


def make_drift_alert(ks_stat: float, p_value: float) -> Alert:
    """Build an alert for a detected data drift event."""
    return Alert(
        severity="warning",
        message=f"Data drift detected: KS={ks_stat:.4f}, p={p_value:.4f}",
        source="drift-monitor",
        tags=["drift"],
        metadata={"ks_statistic": ks_stat, "p_value": p_value},
    )


__all__ = [
    "Alert",
    "AlertQueue",
    "make_anomaly_alert",
    "make_drift_alert",
    "severity_rank",
]
