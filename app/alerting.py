"""Threshold-based alerting engine for metric monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

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
    _last_fired: Optional[float] = field(default=None, repr=False, compare=False)

    _OPS: Dict[str, Callable[[float, float], bool]] = field(
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

    def evaluate(self, value: float, now: float) -> Optional[Alert]:
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
        msg = (
            f"[{self.severity.value.upper()}] {self.name}: "
            f"{self.metric}={value} {self.comparison} {self.threshold}"
        )
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

    def __init__(self, handlers: Optional[List[Callable[[Alert], None]]] = None) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._handlers: List[Callable[[Alert], None]] = handlers or []
        self._fired: List[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alerting rule."""
        self._rules[rule.name] = rule
        logger.debug("Registered alert rule '%s'", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found."""
        return self._rules.pop(name, None) is not None

    def evaluate_all(self, metrics: Dict[str, float], now: Optional[float] = None) -> List[Alert]:
        """Evaluate all registered rules against a metrics snapshot.

        Args:
            metrics: Dict mapping metric name to current value.
            now: Override timestamp (for testing).

        Returns:
            List of alerts that fired.
        """
        import time as _time
        ts = now if now is not None else _time.monotonic()
        fired: List[Alert] = []
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
    def history(self) -> List[Alert]:
        """Return all alerts fired since creation."""
        return list(self._fired)


__all__ = ["Alert", "AlertManager", "AlertRule", "Severity"]
