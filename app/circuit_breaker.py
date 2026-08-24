"""Circuit breaker implementation for protecting downstream calls."""

from __future__ import annotations

import functools
import logging
import time
from enum import Enum
from typing import Callable, Sequence, Type

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
        expected_exceptions: Sequence[Type[Exception]] = (Exception,),
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
        if self._state is CircuitState.OPEN:
            if self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.recovery_timeout:
                    logger.info("Circuit entering HALF_OPEN after %.1fs", elapsed)
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable, *args, **kwargs):
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
            raise CircuitOpenError(
                f"Circuit is OPEN; calls blocked for {self.recovery_timeout}s"
            )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            logger.info("Circuit CLOSED after successful probe")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._state is CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            logger.warning(
                "Circuit OPEN after %d failures", self._failure_count
            )
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


__all__ = ["CircuitBreaker", "CircuitOpenError", "CircuitState"]
