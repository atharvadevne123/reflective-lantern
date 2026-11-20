"""Simple in-memory telemetry counters for Risk-Sense."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int] = {}
_gauges: dict[str, float] = {}


def increment(key: str, amount: int = 1) -> None:
    """Increment a named counter.

    Args:
        key: Counter key name.
        amount: Amount to increment by (default 1).
    """
    with _lock:
        _counters[key] = _counters.get(key, 0) + amount


def get_counter(key: str) -> int:
    """Return the current value of a counter.

    Args:
        key: Counter key name.

    Returns:
        Current counter value, or 0 if not set.
    """
    with _lock:
        return _counters.get(key, 0)


def set_gauge(key: str, value: float) -> None:
    """Set a named gauge to a specific value.

    Args:
        key: Gauge key name.
        value: Value to set.
    """
    with _lock:
        _gauges[key] = value


def get_gauge(key: str) -> float:
    """Return the current value of a gauge.

    Args:
        key: Gauge key name.

    Returns:
        Current gauge value, or 0.0 if not set.
    """
    with _lock:
        return _gauges.get(key, 0.0)


def reset_all() -> None:
    """Reset all counters and gauges (for testing)."""
    with _lock:
        _counters.clear()
        _gauges.clear()


def snapshot() -> dict[str, Any]:
    """Return a copy of all telemetry state.

    Returns:
        Dict with 'counters' and 'gauges' sub-dicts.
    """
    with _lock:
        return {
            "counters": dict(_counters),
            "gauges": dict(_gauges),
        }


def record_prediction(default_probability: float) -> None:
    """Record a prediction event for telemetry.

    Args:
        default_probability: The predicted default probability (0-1).
    """
    increment("predictions_total")
    if default_probability >= 0.5:
        increment("high_risk_predictions")
    elif default_probability < 0.2:
        increment("low_risk_predictions")

    total = get_counter("predictions_total")
    current_avg = get_gauge("avg_default_probability")
    new_avg = (current_avg * (total - 1) + default_probability) / total
    set_gauge("avg_default_probability", new_avg)
    logger.debug("Telemetry recorded: total=%d avg=%.4f", total, new_avg)
