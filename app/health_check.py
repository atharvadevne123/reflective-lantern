"""Composable health-check framework for readiness and liveness probes.

Allows registering named checks, running them in sequence, and aggregating
results into an overall status suitable for /health endpoints.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "CheckResult",
    "HealthStatus",
    "HealthRegistry",
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
    results: List[CheckResult]

    @property
    def failed(self) -> List[CheckResult]:
        """Return only the failing checks."""
        return [r for r in self.results if not r.healthy]


CheckFn = Callable[[], CheckResult]


class HealthRegistry:
    """Registry that runs named health checks and aggregates results."""

    def __init__(self) -> None:
        """Initialise the registry with an empty check dictionary."""
        self._checks: Dict[str, CheckFn] = {}

    def register(self, name: str, fn: CheckFn) -> None:
        """Register *fn* under *name*."""
        self._checks[name] = fn

    def unregister(self, name: str) -> None:
        """Remove the check registered under *name*."""
        self._checks.pop(name, None)

    def run(self) -> HealthStatus:
        """Execute all registered checks and return an aggregated status."""
        results: List[CheckResult] = []
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
        """Return the number of registered health checks."""
        return len(self._checks)


_default_registry = HealthRegistry()


def check(name: str, registry: Optional[HealthRegistry] = None) -> Callable[[CheckFn], CheckFn]:
    """Decorator that registers a function as a named health check."""
    reg = registry or _default_registry

    def decorator(fn: CheckFn) -> CheckFn:
        """Register *fn* and return it unchanged."""
        reg.register(name, fn)
        return fn

    return decorator
