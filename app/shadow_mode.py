"""Shadow mode execution for safe comparison of model versions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Outcome of a shadow execution for one request.

    Attributes:
        primary_result: Return value from the primary handler.
        shadow_result: Return value from the shadow handler (None if failed).
        shadow_error: Exception from shadow handler, if any.
        primary_latency_ms: Milliseconds for the primary call.
        shadow_latency_ms: Milliseconds for the shadow call (or 0 on error).
        matched: Whether primary and shadow results are equal.
    """

    primary_result: Any
    shadow_result: Any | None
    shadow_error: Exception | None
    primary_latency_ms: float
    shadow_latency_ms: float
    matched: bool


class ShadowRunner:
    """Runs a primary handler and a shadow handler for every request.

    The shadow handler's result does not affect the response; errors are
    captured and logged.  Comparison statistics are accumulated for analysis.

    Args:
        primary: Primary callable (result returned to caller).
        shadow: Shadow callable (run for comparison only).
        comparer: Optional callable that decides if two results match;
            defaults to equality comparison.
    """

    def __init__(
        self,
        primary: Callable,
        shadow: Callable,
        comparer: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self.primary = primary
        self.shadow = shadow
        self.comparer = comparer or (lambda a, b: a == b)
        self._results: list[ShadowResult] = []

    def call(self, *args, **kwargs) -> Any:
        """Execute primary and shadow handlers, return primary result.

        Args:
            *args: Forwarded to both handlers.
            **kwargs: Forwarded to both handlers.

        Returns:
            Result of the primary handler.
        """
        t0 = time.perf_counter()
        primary_result = self.primary(*args, **kwargs)
        primary_ms = (time.perf_counter() - t0) * 1000

        shadow_result = None
        shadow_error = None
        shadow_ms = 0.0
        try:
            t1 = time.perf_counter()
            shadow_result = self.shadow(*args, **kwargs)
            shadow_ms = (time.perf_counter() - t1) * 1000
        except Exception as exc:
            shadow_error = exc
            logger.warning("Shadow handler error: %s", exc)

        try:
            matched = shadow_error is None and self.comparer(primary_result, shadow_result)
        except Exception:
            matched = False

        result = ShadowResult(
            primary_result=primary_result,
            shadow_result=shadow_result,
            shadow_error=shadow_error,
            primary_latency_ms=primary_ms,
            shadow_latency_ms=shadow_ms,
            matched=matched,
        )
        self._results.append(result)
        if not matched:
            logger.info(
                "Shadow mismatch: primary=%r shadow=%r",
                primary_result,
                shadow_result,
            )
        return primary_result

    def stats(self) -> dict[str, Any]:
        """Return aggregate comparison statistics.

        Returns:
            Dict with total, matched, mismatched, error counts and match rate.
        """
        total = len(self._results)
        matched = sum(1 for r in self._results if r.matched)
        errors = sum(1 for r in self._results if r.shadow_error is not None)
        return {
            "total": total,
            "matched": matched,
            "mismatched": total - matched - errors,
            "errors": errors,
            "match_rate": matched / total if total else 0.0,
        }

    def clear(self) -> None:
        """Reset accumulated results."""
        self._results.clear()


__all__ = ["ShadowResult", "ShadowRunner"]
