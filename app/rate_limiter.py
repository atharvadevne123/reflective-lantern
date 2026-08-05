"""Token-bucket rate limiter for per-client request throttling."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
            logger.debug("rate_limit: client=%r throttled tokens=%.2f", client_key, bucket.tokens)
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


def make_rate_limiter(capacity: float = 60.0, rate_per_second: float = 1.0) -> TokenBucketRateLimiter:
    """Factory returning a new :class:`TokenBucketRateLimiter` with defaults."""
    return TokenBucketRateLimiter(capacity=capacity, rate_per_second=rate_per_second)


__all__ = [
    "TokenBucketRateLimiter",
    "make_rate_limiter",
]


def limiter_utilization(limiter: "TokenBucketRateLimiter", client_key: str = "default") -> float:
    """Return the fraction of token capacity currently consumed for a client.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: The client identifier. Default "default".

    Returns:
        Float in [0, 1] where 0 = full capacity available, 1 = fully consumed.
    """
    remaining = limiter.remaining_tokens(client_key)
    capacity = limiter._capacity
    consumed = capacity - remaining
    return round(max(0.0, min(1.0, consumed / capacity)), 4)


def reset_limiter(limiter: "TokenBucketRateLimiter", client_key: str = "default") -> None:
    """Reset a client's token bucket to full capacity.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: The client identifier. Default "default".
    """
    limiter.reset(client_key)


def is_rate_limited(limiter: "TokenBucketRateLimiter", client_key: str = "default") -> bool:
    """Check whether a client's next request would be rate-limited WITHOUT consuming tokens.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: The client identifier. Default "default".

    Returns:
        True if the next is_allowed call would be denied, else False.
    """
    return limiter.remaining_tokens(client_key) < 1.0


def make_strict_limiter(max_per_second: float) -> "TokenBucketRateLimiter":
    """Create a rate limiter with capacity = max_per_second (burst of 1 second).

    Args:
        max_per_second: Allowed requests per second.

    Returns:
        TokenBucketRateLimiter configured for the given rate.
    """
    return make_rate_limiter(capacity=max_per_second, rate_per_second=max_per_second)
