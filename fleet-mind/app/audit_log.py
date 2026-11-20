"""Append-only audit log for security-sensitive API operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "/tmp/fleet_mind_audit.jsonl"))


def _write_entry(entry: dict[str, Any]) -> None:
    """Append one JSON entry to the audit log file.

    Args:
        entry: Dict to serialise and append.
    """
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.error("Audit log write failed: %s", exc)


def log_prediction_request(
    vehicle_id: str,
    client_ip: str,
    correlation_id: str,
    maintenance_needed: bool,
) -> None:
    """Record an inbound prediction request.

    Args:
        vehicle_id: Identifier of the vehicle being assessed.
        client_ip: IP address of the API caller.
        correlation_id: Request correlation identifier.
        maintenance_needed: Whether the model flagged maintenance.
    """
    _write_entry(
        {
            "event": "prediction_request",
            "ts": datetime.utcnow().isoformat(),
            "vehicle_id": vehicle_id,
            "client_ip": client_ip,
            "correlation_id": correlation_id,
            "maintenance_needed": maintenance_needed,
        }
    )


def log_retrain_event(
    triggered_by: str,
    metrics: dict[str, float],
    correlation_id: str,
) -> None:
    """Record a model retraining event.

    Args:
        triggered_by: IP or identifier of who triggered the retrain.
        metrics: Post-training evaluation metrics.
        correlation_id: Request correlation identifier.
    """
    _write_entry(
        {
            "event": "model_retrain",
            "ts": datetime.utcnow().isoformat(),
            "triggered_by": triggered_by,
            "metrics": metrics,
            "correlation_id": correlation_id,
        }
    )


def recent_audit_entries(n: int = 50) -> list[dict[str, Any]]:
    """Return the last n audit log entries.

    Args:
        n: Maximum number of entries to return.

    Returns:
        List of parsed audit entry dicts, most recent last.
    """
    if not _AUDIT_LOG_PATH.exists():
        return []
    try:
        lines = _AUDIT_LOG_PATH.read_text().strip().splitlines()
        return [json.loads(line) for line in lines[-n:]]
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read audit log: %s", exc)
        return []


def audit_entry_count() -> int:
    """Return the total number of lines in the audit log.

    Returns:
        Line count, or 0 if the log does not exist or is empty.
    """
    if not _AUDIT_LOG_PATH.exists():
        return 0
    try:
        return sum(1 for _ in _AUDIT_LOG_PATH.open())
    except OSError as exc:
        logger.error("Failed to count audit log entries: %s", exc)
        return 0


def filter_audit_entries(event_type: str, n: int = 100) -> list[dict[str, Any]]:
    """Return recent audit entries filtered by event type.

    Args:
        event_type: Event type string to match (e.g. 'prediction', 'model_retrain').
        n: Maximum number of entries to scan from the tail of the log.

    Returns:
        List of matching entry dicts, most recent last.
    """
    entries = recent_audit_entries(n)
    return [e for e in entries if e.get("event") == event_type]
