"""Dynamic alert threshold management for fleet telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass
class Threshold:
    metric: str
    warning: float
    critical: float
    direction: str = "high"


@dataclass
class ThresholdStore:
    _thresholds: dict[str, Threshold] = field(default_factory=dict)

    def set(self, threshold: Threshold) -> None:
        self._thresholds[threshold.metric] = threshold

    def get(self, metric: str) -> Threshold | None:
        return self._thresholds.get(metric)

    def all_metrics(self) -> list[str]:
        return list(self._thresholds.keys())


def evaluate(value: float, threshold: Threshold) -> str:
    """Return 'ok', 'warning', or 'critical' for a telemetry value."""
    if threshold.direction == "high":
        if value >= threshold.critical:
            return "critical"
        if value >= threshold.warning:
            return "warning"
        return "ok"
    else:
        if value <= threshold.critical:
            return "critical"
        if value <= threshold.warning:
            return "warning"
        return "ok"


def adaptive_thresholds(
    historical: list[float],
    warning_sigma: float = 2.0,
    critical_sigma: float = 3.0,
    direction: str = "high",
) -> tuple[float, float]:
    """Derive warning and critical thresholds from historical data using sigma multiples."""
    if len(historical) < 2:
        raise ValueError("Need at least 2 historical values")
    mu = mean(historical)
    sigma = stdev(historical)
    if direction == "high":
        return mu + warning_sigma * sigma, mu + critical_sigma * sigma
    return mu - warning_sigma * sigma, mu - critical_sigma * sigma


def batch_evaluate(
    values: dict[str, float],
    store: ThresholdStore,
) -> dict[str, str]:
    """Evaluate multiple metric values and return a status dict."""
    result: dict[str, str] = {}
    for metric, value in values.items():
        threshold = store.get(metric)
        result[metric] = evaluate(value, threshold) if threshold else "unknown"
    return result
