"""Token-bucket rate limiter for fine-grained throughput control."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Tokens refill at ``rate`` per second up to ``capacity``.  Each call to
    :meth:`consume` removes ``tokens`` from the bucket; if insufficient tokens
    are available it returns False immediately (non-blocking).

    Args:
        capacity: Maximum token count (burst size).
        rate: Tokens added per second.
    """

    capacity: float
    rate: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + added)
        self._last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were available and consumed, False otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            logger.debug(
                "Rate limit: requested %.1f tokens, only %.1f available",
                tokens,
                self._tokens,
            )
            return False

    def wait_and_consume(self, tokens: float = 1.0, timeout: float = 5.0) -> bool:
        """Block until tokens are available or timeout elapses.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait.

        Returns:
            True if tokens were consumed within timeout, False otherwise.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.consume(tokens):
                return True
            time.sleep(min(tokens / self.rate, 0.05))
        return False

    @property
    def available(self) -> float:
        """Return current token count after a refill."""
        with self._lock:
            self._refill()
            return self._tokens


class PerKeyTokenBucket:
    """Maintains a separate TokenBucket per key (e.g. per client IP).

    Args:
        capacity: Burst capacity shared by each new bucket.
        rate: Refill rate shared by each new bucket.
    """

    def __init__(self, capacity: float, rate: float) -> None:
        self.capacity = capacity
        self.rate = rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, tokens: float = 1.0) -> bool:
        """Consume tokens from the bucket for key, creating it on first use."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(self.capacity, self.rate)
            bucket = self._buckets[key]
        return bucket.consume(tokens)

    def bucket_count(self) -> int:
        """Return number of distinct keys tracked."""
        return len(self._buckets)


__all__ = ["PerKeyTokenBucket", "TokenBucket"]
