"""Token-bucket rate limiter for fine-grained throughput control.

Implements the token-bucket algorithm for smooth rate limiting:

- Tokens accumulate at a fixed ``rate`` per second up to ``capacity``.
- Each :meth:`~TokenBucket.consume` call removes tokens non-blockingly;
  callers that need blocking behaviour can use
  :meth:`~TokenBucket.wait_and_consume`.
- :class:`PerKeyTokenBucket` maintains an independent bucket per key (e.g.
  per client IP or user ID), useful for per-tenant rate limiting.

Both classes are thread-safe via :class:`threading.Lock`.

Example::

    from app.token_bucket import TokenBucket

    limiter = TokenBucket(capacity=10, rate=5.0)  # 5 req/s, burst of 10

    if limiter.consume():
        process_request()
    else:
        raise RateLimitExceeded()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

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
        """Validate parameters and initialise mutable state fields."""
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Compute elapsed time since last refill and add tokens accordingly.

        Caps accumulated tokens at :attr:`capacity` to prevent unbounded growth
        during long idle periods.  Must be called while holding :attr:`_lock`.
        """
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

    @property
    def fill_ratio(self) -> float:
        """Return the fraction of capacity currently available (0.0–1.0)."""
        return self.available / self.capacity


class PerKeyTokenBucket:
    """Maintains a separate TokenBucket per key (e.g. per client IP).

    Args:
        capacity: Burst capacity shared by each new bucket.
        rate: Refill rate shared by each new bucket.
    """

    def __init__(self, capacity: float, rate: float) -> None:
        """Initialise the keyed limiter with shared capacity and rate.

        Args:
            capacity: Bucket capacity (burst size) applied to every key.
            rate: Refill rate in tokens per second applied to every key.
        """
        self.capacity = capacity
        self.rate = rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, tokens: float = 1.0) -> bool:
        """Consume tokens from the bucket for *key*, creating it on first use.

        Args:
            key: Client or resource identifier (e.g. IP address, user ID).
            tokens: Number of tokens to consume (default 1).

        Returns:
            True if the bucket for *key* had sufficient tokens; False otherwise.
        """
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(self.capacity, self.rate)
            bucket = self._buckets[key]
        return bucket.consume(tokens)

    def bucket_count(self) -> int:
        """Return the number of distinct keys currently tracked.

        Returns:
            Integer count of keys that have had at least one :meth:`consume` call.
        """
        return len(self._buckets)


__all__ = ["PerKeyTokenBucket", "TokenBucket"]
