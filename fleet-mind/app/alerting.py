"""Alerting utilities — emit structured log alerts for critical fleet anomalies."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds that warrant an immediate alert regardless of model output
_CRITICAL_THRESHOLDS: dict[str, tuple[float, str]] = {
    "engine_temp_c": (115.0, "CRITICAL_ENGINE_OVERHEAT"),
    "oil_pressure_bar": (0.8, "CRITICAL_OIL_PRESSURE_LOW"),
    "brake_pad_mm": (1.5, "CRITICAL_BRAKE_WORN"),
    "battery_voltage_v": (10.0, "CRITICAL_BATTERY_LOW"),
    "coolant_temp_c": (110.0, "CRITICAL_COOLANT_OVERHEAT"),
    "transmission_temp_c": (130.0, "CRITICAL_TRANSMISSION_OVERHEAT"),
}


def check_critical_thresholds(
    vehicle_id: str,
    reading: dict[str, float],
) -> list[dict[str, Any]]:
    """Check raw telemetry values against hard safety thresholds.

    Critical readings bypass the ML model and generate immediate log alerts.

    Args:
        vehicle_id: The identifier of the vehicle being checked.
        reading: Raw telemetry mapping from feature name to float value.

    Returns:
        List of alert dicts, each with 'vehicle_id', 'alert_type', 'field',
        'value', and 'threshold' keys. Empty list if all readings are safe.
    """
    alerts: list[dict[str, Any]] = []

    for field, (threshold, alert_type) in _CRITICAL_THRESHOLDS.items():
        value = reading.get(field)
        if value is None:
            continue
        if value <= threshold:
            alert = {
                "vehicle_id": vehicle_id,
                "alert_type": alert_type,
                "field": field,
                "value": value,
                "threshold": threshold,
            }
            alerts.append(alert)
            logger.critical(
                "FLEET ALERT %s: vehicle=%s %s=%.2f (threshold=%.2f)",
                alert_type,
                vehicle_id,
                field,
                value,
                threshold,
            )

    return alerts


def alert_on_drift(feature: str, ks_statistic: float, p_value: float) -> None:
    """Emit a structured log alert when feature drift is detected.

    Args:
        feature: Name of the drifted feature.
        ks_statistic: KS test statistic.
        p_value: KS test p-value.
    """
    logger.warning(
        "DRIFT ALERT: feature=%s ks_statistic=%.6f p_value=%.6f",
        feature,
        ks_statistic,
        p_value,
    )
