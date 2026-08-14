"""Alert and notification helpers for energy anomaly events."""

from __future__ import annotations

import logging
import time as _time
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
    created_at: float = field(default_factory=_time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialise this alert to a plain dictionary."""
        return {
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
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
        logger.debug(
            "AlertQueue.push: severity=%s source=%s queue_size=%d", alert.severity, alert.source, len(self._alerts)
        )

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
        """Return the current number of queued alerts."""
        return len(self._alerts)

    @property
    def alerts(self) -> list[Alert]:
        """Return a shallow copy of the current alert list."""
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
    "alert_summary",
    "deduplicate_alerts",
    "filter_alerts_by_severity",
    "group_alerts_by_severity",
    "highest_severity",
    "make_anomaly_alert",
    "make_drift_alert",
    "make_performance_alert",
    "mute_alert",
    "severity_rank",
]


def alert_summary(alerts: list[Alert]) -> dict[str, int]:
    """Return a count of alerts grouped by severity.

    Args:
        alerts: List of :class:`Alert` instances.

    Returns:
        Dict mapping severity name to count (info, warning, critical).
    """
    summary: dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
    for a in alerts:
        key = a.severity.lower()
        if key in summary:
            summary[key] += 1
    return summary


def highest_severity(alerts: list[Alert]) -> str:
    """Return the highest severity level among *alerts*.

    Args:
        alerts: List of :class:`Alert` instances.

    Returns:
        Severity string ('info', 'warning', 'critical'), or 'none' if empty.
    """
    if not alerts:
        return "none"
    return max(alerts, key=lambda a: severity_rank(a.severity)).severity


def filter_alerts_by_severity(alerts: list[Alert], min_severity: str) -> list[Alert]:
    """Return only alerts at or above the minimum severity level.

    Args:
        alerts: List of Alert objects.
        min_severity: Minimum severity string ("low", "medium", "high", "critical").

    Returns:
        Filtered list of alerts.
    """
    min_rank = severity_rank(min_severity)
    return [a for a in alerts if severity_rank(a.severity) >= min_rank]


def deduplicate_alerts(alerts: list[Alert]) -> list[Alert]:
    """Remove duplicate alerts that share the same message.

    Args:
        alerts: List of Alert objects.

    Returns:
        List with unique messages, preserving first occurrence order.
    """
    seen: set[str] = set()
    result = []
    for alert in alerts:
        if alert.message not in seen:
            seen.add(alert.message)
            result.append(alert)
    return result


def format_alert_text(alert: Alert) -> str:
    """Format an Alert as a human-readable text line.

    Args:
        alert: Alert object with severity, message, timestamp attributes.

    Returns:
        Formatted string like "[HIGH] Drift detected - 2024-01-01T12:00:00".
    """
    ts = getattr(alert, "timestamp", "")
    return f"[{alert.severity.upper()}] {alert.message} - {ts}"


def batch_alerts_by_severity(alerts: list[Alert]) -> dict[str, list[Alert]]:
    """Group alerts into batches by severity level.

    Args:
        alerts: List of Alert objects.

    Returns:
        Dict mapping severity -> list of matching alerts.
    """
    batches: dict[str, list[Alert]] = {}
    for alert in alerts:
        batches.setdefault(alert.severity, []).append(alert)
    return batches


def make_performance_alert(
    metric: str,
    current_value: float,
    threshold: float,
    is_critical: bool = False,
) -> Alert:
    """Build an alert for a model or system performance threshold breach.

    Args:
        metric: Name of the metric that breached the threshold.
        current_value: Observed metric value.
        threshold: Alert threshold.
        is_critical: If True, sets severity to 'critical'; otherwise 'warning'.

    Returns:
        A populated :class:`Alert` instance.
    """
    severity = "critical" if is_critical else "warning"
    return Alert(
        severity=severity,
        message=f"Performance alert: {metric}={current_value:.4f} exceeds threshold {threshold:.4f}",
        source="performance-monitor",
        tags=["performance", metric],
        metadata={"metric": metric, "current_value": current_value, "threshold": threshold},
    )


def group_alerts_by_severity(alerts: list[Alert]) -> dict[str, list[Alert]]:
    """Group *alerts* by their severity level.

    Args:
        alerts: List of Alert objects.

    Returns:
        Dict mapping severity strings to lists of matching alerts.
        Always contains keys 'info', 'warning', 'critical'.
    """
    groups: dict[str, list[Alert]] = {"info": [], "warning": [], "critical": []}
    for alert in alerts:
        key = alert.severity.lower()
        groups.setdefault(key, []).append(alert)
    return groups


def deduplicate_alerts_by_window(alerts: list[Alert], window_seconds: float = 300.0) -> list[Alert]:
    """Remove consecutive duplicate alerts within a time window.

    An alert is a duplicate if it has the same source and message as the
    previous alert from that source and arrived within *window_seconds*.

    Args:
        alerts: Time-ordered list of Alert objects.
        window_seconds: Maximum gap (in seconds) within which a repeat is
            suppressed.

    Returns:
        Filtered list with consecutive duplicates removed.
    """
    seen: dict[str, tuple[str, float]] = {}  # source -> (message, timestamp)
    result: list[Alert] = []
    now = _time.time()
    for alert in alerts:
        key = alert.source
        prev_msg, prev_ts = seen.get(key, ("", 0.0))
        age = now - prev_ts
        if alert.message == prev_msg and age < window_seconds:
            continue
        seen[key] = (alert.message, now)
        result.append(alert)
    return result


def top_alerts(alerts: list[Alert], n: int = 5) -> list[Alert]:
    """Return the *n* most severe alerts, breaking ties by insertion order (latest first).

    Args:
        alerts: List of Alert objects.
        n: Maximum number to return.

    Returns:
        Up to *n* alerts sorted by severity (highest first), then by recency
        (later items in *alerts* are treated as more recent).
    """
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    indexed = list(enumerate(alerts))
    sorted_alerts = sorted(
        indexed,
        key=lambda ia: (severity_order.get(ia[1].severity.lower(), 99), -ia[0]),
    )
    return [a for _, a in sorted_alerts[:n]]


def mute_alert(alert: Alert, reason: str = "") -> Alert:
    """Return a copy of *alert* tagged as muted.

    The returned alert has 'muted' added to its tags list and, if *reason*
    is provided, a 'mute_reason' key in its metadata.

    Args:
        alert: The Alert instance to mute.
        reason: Optional human-readable reason for muting.

    Returns:
        A new Alert with updated tags and metadata.
    """
    new_tags = [*list(alert.tags), "muted"]
    new_meta = dict(alert.metadata)
    if reason:
        new_meta["mute_reason"] = reason
    return Alert(
        severity=alert.severity,
        message=alert.message,
        source=alert.source,
        tags=new_tags,
        metadata=new_meta,
    )
