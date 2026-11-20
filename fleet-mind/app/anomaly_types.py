"""Anomaly type definitions and multi-label classification for fleet telemetry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyRule:
    """A named, threshold-based anomaly detection rule.

    Attributes:
        name: Machine-readable anomaly identifier.
        description: Human-readable description.
        field: Telemetry feature this rule inspects.
        threshold: The value at which the anomaly is triggered.
        direction: 'above' if value > threshold triggers, 'below' if value < threshold triggers.
        severity: One of 'critical', 'warning', or 'info'.
    """

    name: str
    description: str
    field: str
    threshold: float
    direction: str  # 'above' | 'below'
    severity: str  # 'critical' | 'warning' | 'info'

    def check(self, reading: dict[str, float]) -> bool:
        """Return True if this rule fires for the given reading.

        Args:
            reading: Dict mapping feature name to observed value.

        Returns:
            True if the anomaly condition is met.
        """
        value = reading.get(self.field)
        if value is None:
            return False
        if self.direction == "above":
            return value > self.threshold
        return value < self.threshold


ANOMALY_RULES: list[AnomalyRule] = [
    AnomalyRule("engine_overheat", "Engine temperature critically high", "engine_temp_c", 105.0, "above", "critical"),
    AnomalyRule("engine_warm", "Engine temperature elevated", "engine_temp_c", 95.0, "above", "warning"),
    AnomalyRule("oil_pressure_critical", "Oil pressure dangerously low", "oil_pressure_bar", 1.0, "below", "critical"),
    AnomalyRule("oil_pressure_low", "Oil pressure below normal", "oil_pressure_bar", 2.0, "below", "warning"),
    AnomalyRule("brake_critical", "Brake pads at minimum safe thickness", "brake_pad_mm", 2.0, "below", "critical"),
    AnomalyRule("brake_worn", "Brake pads wearing thin", "brake_pad_mm", 3.5, "below", "warning"),
    AnomalyRule("battery_critical", "Battery voltage critically low", "battery_voltage_v", 10.5, "below", "critical"),
    AnomalyRule("battery_low", "Battery voltage below normal", "battery_voltage_v", 12.0, "below", "warning"),
    AnomalyRule("tire_pressure_low", "Tyre pressure below safe minimum", "tire_pressure_bar", 1.8, "below", "warning"),
    AnomalyRule("transmission_overheat", "Transmission temperature critical", "transmission_temp_c", 120.0, "above", "critical"),
    AnomalyRule("coolant_overheat", "Coolant temperature critical", "coolant_temp_c", 105.0, "above", "critical"),
]


def classify_anomalies(reading: dict[str, float]) -> list[dict[str, Any]]:
    """Return all anomaly rules that fire for the given reading.

    Args:
        reading: Dict mapping feature name to observed float value.

    Returns:
        List of dicts with 'name', 'description', 'severity', 'field', 'value',
        and 'threshold' for each triggered rule, sorted by severity (critical first).
    """
    triggered = []
    for rule in ANOMALY_RULES:
        if rule.check(reading):
            triggered.append({
                "name": rule.name,
                "description": rule.description,
                "severity": rule.severity,
                "field": rule.field,
                "value": reading.get(rule.field),
                "threshold": rule.threshold,
            })

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    triggered.sort(key=lambda x: severity_order.get(x["severity"], 9))

    if triggered:
        logger.info("Anomalies detected: %s", [a["name"] for a in triggered])

    return triggered
