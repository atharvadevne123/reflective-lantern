"""Composable health-check framework for readiness and liveness probes.

Allows registering named checks, running them in sequence, and aggregating
results into an overall status suitable for /health endpoints.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "CheckResult",
    "HealthRegistry",
    "HealthStatus",
    "check",
]


@dataclass
class CheckResult:
    """Outcome of a single health check."""

    name: str
    healthy: bool
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Aggregated health report."""

    healthy: bool
    results: list[CheckResult]

    @property
    def failed(self) -> list[CheckResult]:
        """Return only the failing checks."""
        return [r for r in self.results if not r.healthy]


CheckFn = Callable[[], CheckResult]


class HealthRegistry:
    """Registry that runs named health checks and aggregates results."""

    def __init__(self) -> None:
        self._checks: dict[str, CheckFn] = {}

    def register(self, name: str, fn: CheckFn) -> None:
        """Register *fn* under *name*."""
        self._checks[name] = fn

    def unregister(self, name: str) -> None:
        """Remove the check registered under *name*."""
        self._checks.pop(name, None)

    def run(self) -> HealthStatus:
        """Execute all registered checks and return an aggregated status."""
        results: list[CheckResult] = []
        for name, fn in self._checks.items():
            try:
                result = fn()
            except Exception:
                result = CheckResult(
                    name=name,
                    healthy=False,
                    message=traceback.format_exc(limit=3),
                )
            results.append(result)
        overall = all(r.healthy for r in results)
        return HealthStatus(healthy=overall, results=results)

    def __len__(self) -> int:
        return len(self._checks)


_default_registry = HealthRegistry()


def check(name: str, registry: HealthRegistry | None = None) -> Callable[[CheckFn], CheckFn]:
    """Decorator that registers a function as a named health check."""
    reg = _default_registry if registry is None else registry

    def decorator(fn: CheckFn) -> CheckFn:
        reg.register(name, fn)
        return fn

    return decorator
