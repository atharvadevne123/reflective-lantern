"""Circuit breaker implementation for protecting downstream calls."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Sequence
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Tracks consecutive failures and opens the circuit to fast-fail callers.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait in OPEN before trying HALF_OPEN.
        expected_exceptions: Exception types that count as failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: Sequence[type[Exception]] = (Exception,),
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = tuple(expected_exceptions)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Return current circuit state, transitioning OPEN -> HALF_OPEN when ready."""
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                logger.info("Circuit entering HALF_OPEN after %.1fs", elapsed)
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable, *args: object, **kwargs: object) -> object:
        """Execute func through the circuit breaker.

        Args:
            func: The callable to protect.
            *args: Positional arguments forwarded to func.
            **kwargs: Keyword arguments forwarded to func.

        Raises:
            CircuitOpenError: When the circuit is OPEN.
            Exception: Whatever func raises when the circuit is CLOSED or HALF_OPEN.
        """
        if self.state is CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit is OPEN; calls blocked for {self.recovery_timeout}s")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        """Record a successful call, closing the circuit if it was half-open.

        Resets :attr:`_failure_count` to zero and clears :attr:`_opened_at`.
        Transitions HALF_OPEN → CLOSED and leaves CLOSED unchanged.
        """
        if self._state is CircuitState.HALF_OPEN:
            logger.info("Circuit CLOSED after successful probe")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        """Record a failed call, opening the circuit when threshold is exceeded.

        Increments :attr:`_failure_count`. Transitions to OPEN when the count
        reaches :attr:`failure_threshold`, or immediately when HALF_OPEN (one
        bad probe re-opens without waiting for the threshold).
        """
        self._failure_count += 1
        if self._state is CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            logger.warning("Circuit OPEN after %d failures", self._failure_count)
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def __call__(self, func: Callable) -> Callable:
        """Use this :class:`CircuitBreaker` instance as a function decorator.

        Args:
            func: The function to wrap with circuit-breaker protection.

        Returns:
            A wrapper that calls *func* through :meth:`call` on every invocation.
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Forward the call through the circuit breaker."""
            return self.call(func, *args, **kwargs)

        return wrapper


__all__ = ["CircuitBreaker", "CircuitOpenError", "CircuitState"]
