"""Token-bucket rate limiter for per-client request throttling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter.

    Each client key gets an independent bucket. Tokens refill at
    *rate_per_second* up to *capacity*. A request consumes one token.
    """

    def __init__(self, capacity: float = 10.0, rate_per_second: float = 1.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._capacity = capacity
        self._rate = rate_per_second
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    def _refill(self, bucket: _Bucket) -> None:
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)
        bucket.last_refill = now

    def is_allowed(self, client_key: str) -> bool:
        """Return True and consume one token if the client is within rate limit."""
        with self._lock:
            if client_key not in self._buckets:
                self._buckets[client_key] = _Bucket(tokens=self._capacity)
            bucket = self._buckets[client_key]
            self._refill(bucket)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def remaining_tokens(self, client_key: str) -> float:
        """Return current token count for *client_key* after a refill."""
        with self._lock:
            if client_key not in self._buckets:
                return self._capacity
            bucket = self._buckets[client_key]
            self._refill(bucket)
            return bucket.tokens

    def reset(self, client_key: str) -> None:
        """Reset the bucket for *client_key* to full capacity."""
        with self._lock:
            self._buckets[client_key] = _Bucket(tokens=self._capacity)

    def clear(self) -> None:
        """Remove all client buckets."""
        with self._lock:
            self._buckets.clear()

    @property
    def client_count(self) -> int:
        """Number of tracked client keys."""
        with self._lock:
            return len(self._buckets)


def make_rate_limiter(
    capacity: float = 60.0, rate_per_second: float = 1.0
) -> TokenBucketRateLimiter:
    """Factory returning a new :class:`TokenBucketRateLimiter` with defaults."""
    return TokenBucketRateLimiter(capacity=capacity, rate_per_second=rate_per_second)


__all__ = [
    "TokenBucketRateLimiter",
    "burst_capacity_fraction",
    "make_rate_limiter",
    "make_strict_limiter",
]


def make_strict_limiter(requests_per_minute: int) -> TokenBucketRateLimiter:
    """Create a rate limiter configured to allow *requests_per_minute* requests.

    Sets capacity equal to requests_per_minute and refill rate to the equivalent
    per-second rate (requests_per_minute / 60).

    Args:
        requests_per_minute: Target request allowance per minute (must be >= 1).

    Returns:
        A configured :class:`TokenBucketRateLimiter` instance.

    Raises:
        ValueError: If *requests_per_minute* is less than 1.
    """
    if requests_per_minute < 1:
        raise ValueError(f"requests_per_minute must be >= 1, got {requests_per_minute}")
    rate = requests_per_minute / 60.0
    return TokenBucketRateLimiter(capacity=float(requests_per_minute), rate_per_second=rate)


def burst_capacity_fraction(limiter: TokenBucketRateLimiter, client_key: str) -> float:
    """Return the current token fill fraction [0.0, 1.0] for *client_key*.

    A value of 1.0 means the bucket is full; 0.0 means it is empty.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.
        client_key: The client identifier to query.

    Returns:
        Float in [0.0, 1.0] representing the fraction of capacity remaining.
    """
    remaining = limiter.remaining_tokens(client_key)
    capacity = limiter._capacity
    if capacity <= 0:
        return 0.0
    return round(min(1.0, remaining / capacity), 6)
