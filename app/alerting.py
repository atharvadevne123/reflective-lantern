"""Threshold-based alerting engine for metric monitoring.

This module provides:

- :class:`AlertRule` — configurable rule evaluated against a named metric with
  support for six comparison operators and a per-rule cooldown period.
- :class:`AlertManager` — registry of rules that evaluates them against a
  metrics snapshot, dispatches fired alerts to registered handlers, and
  maintains a firing history.
- :class:`Alert` — immutable record of a single triggered alert.
- :func:`count_by_severity` — convenience aggregation helper.

Typical usage::

    from app.alerting import AlertManager, AlertRule, Severity

    manager = AlertManager(handlers=[print])
    manager.add_rule(AlertRule("cpu_high", metric="cpu_pct", threshold=90.0))
    alerts = manager.evaluate_all({"cpu_pct": 95.0})
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A triggered alert.

    Attributes:
        name: Rule name that triggered.
        metric: Metric name evaluated.
        value: Observed metric value.
        threshold: Threshold that was breached.
        severity: Severity level.
        message: Human-readable description.
    """

    name: str
    metric: str
    value: float
    threshold: float
    severity: Severity
    message: str


@dataclass
class AlertRule:
    """Configuration for a single alerting rule.

    Attributes:
        name: Unique rule identifier.
        metric: Metric name to watch.
        threshold: Value that, when exceeded, fires the alert.
        severity: Alert severity on breach.
        comparison: One of ``'>'``, ``'>='``, ``'<'``, ``'<='``, ``'=='``.
        cooldown_s: Minimum seconds between repeated alerts for this rule.
    """

    name: str
    metric: str
    threshold: float
    severity: Severity = Severity.WARNING
    comparison: str = ">"
    cooldown_s: float = 60.0
    _last_fired: float | None = field(default=None, repr=False, compare=False)

    _OPS: dict[str, Callable[[float, float], bool]] = field(
        default_factory=lambda: {
            ">": lambda v, t: v > t,
            ">=": lambda v, t: v >= t,
            "<": lambda v, t: v < t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate alert rule configuration after dataclass initialisation."""
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s must be non-negative, got {self.cooldown_s}")

    def evaluate(self, value: float, now: float) -> Alert | None:
        """Evaluate the rule against a metric value.

        Args:
            value: Current metric value.
            now: Current timestamp (monotonic seconds).

        Returns:
            An :class:`Alert` if the rule fires, else None.
        """
        op = self._OPS.get(self.comparison)
        if op is None:
            logger.error("Unknown comparison operator: %s", self.comparison)
            return None
        if not op(value, self.threshold):
            return None
        if self._last_fired is not None and (now - self._last_fired) < self.cooldown_s:
            logger.debug("Rule '%s' in cooldown", self.name)
            return None
        self._last_fired = now
        msg = f"[{self.severity.value.upper()}] {self.name}: {self.metric}={value} {self.comparison} {self.threshold}"
        logger.warning(msg)
        return Alert(
            name=self.name,
            metric=self.metric,
            value=value,
            threshold=self.threshold,
            severity=self.severity,
            message=msg,
        )


class AlertManager:
    """Manages a collection of alert rules and dispatches alerts to handlers.

    Args:
        handlers: Optional list of callables that receive each fired Alert.
    """

    def __init__(self, handlers: list[Callable[[Alert], None]] | None = None) -> None:
        """Initialise the alerting engine with optional alert handlers.

        Args:
            handlers: Callables invoked for every fired alert.
        """
        self._rules: dict[str, AlertRule] = {}
        self._handlers: list[Callable[[Alert], None]] = handlers or []
        self._fired: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alerting rule.

        Args:
            rule: The :class:`AlertRule` to register. Any existing rule with the
                same :attr:`~AlertRule.name` is replaced.
        """
        self._rules[rule.name] = rule
        logger.debug("Registered alert rule '%s'", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: The :attr:`~AlertRule.name` of the rule to remove.

        Returns:
            ``True`` if the rule existed and was removed; ``False`` otherwise.
        """
        return self._rules.pop(name, None) is not None

    def evaluate_all(self, metrics: dict[str, float], now: float | None = None) -> list[Alert]:
        """Evaluate all registered rules against a metrics snapshot.

        Args:
            metrics: Dict mapping metric name to current value.
            now: Override timestamp (for testing).

        Returns:
            List of alerts that fired.
        """
        import time as _time

        ts = now if now is not None else _time.monotonic()
        fired: list[Alert] = []
        for rule in self._rules.values():
            value = metrics.get(rule.metric)
            if value is None:
                continue
            alert = rule.evaluate(value, ts)
            if alert:
                fired.append(alert)
                self._fired.append(alert)
                for h in self._handlers:
                    try:
                        h(alert)
                    except Exception as exc:
                        logger.error("Alert handler failed: %s", exc)
        return fired

    @property
    def history(self) -> list[Alert]:
        """Return all alerts fired since creation.

        Returns:
            A new list snapshot of every :class:`Alert` that was returned by
            :meth:`evaluate_all` since this manager was created.
        """
        return list(self._fired)

    def clear_history(self) -> None:
        """Remove all previously fired alerts from the history list."""
        self._fired.clear()

    def rule_names(self) -> list[str]:
        """Return sorted list of registered rule names."""
        return sorted(self._rules)

    def history_for_metric(self, metric: str) -> list[Alert]:
        """Return all historical alerts for a specific metric name."""
        return [a for a in self._fired if a.metric == metric]

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an additional alert handler."""
        self._handlers.append(handler)


def count_by_severity(alerts: list[Alert]) -> dict[str, int]:
    """Count alerts grouped by severity level.

    Args:
        alerts: List of Alert objects to summarise.

    Returns:
        Dict mapping each severity value string to its count.
        Only severities present in *alerts* are included.
    """
    counts: dict[str, int] = {}
    for alert in alerts:
        key = alert.severity.value
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["Alert", "AlertManager", "AlertRule", "Severity", "count_by_severity"]
