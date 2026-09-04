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
        """Initialise the rate limiter with given capacity and refill rate.

        Args:
            capacity: Maximum number of tokens each bucket can hold.
            rate_per_second: Tokens added per second to each bucket.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._capacity = capacity
        self._rate = rate_per_second
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    def _refill(self, bucket: _Bucket) -> None:
        """Refill *bucket* tokens based on elapsed time since last refill.

        Args:
            bucket: The :class:`_Bucket` to update in-place. Tokens are capped
                at :attr:`_capacity`. Must be called while holding :attr:`_lock`.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)
        bucket.last_refill = now

    def is_allowed(self, client_key: str) -> bool:
        """Return True and consume one token if the client is within rate limit.

        Args:
            client_key: Identifier for the client (e.g. IP address, user ID).

        Returns:
            True when a token was available and consumed; False when the bucket
            is empty and the request should be throttled.
        """
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
        """Return current token count for *client_key* after a refill.

        Args:
            client_key: Client identifier.

        Returns:
            Float token count, up to :attr:`_capacity`. Returns the full
            capacity when the key has never been seen.
        """
        with self._lock:
            if client_key not in self._buckets:
                return self._capacity
            bucket = self._buckets[client_key]
            self._refill(bucket)
            return bucket.tokens

    def reset(self, client_key: str) -> None:
        """Reset the bucket for *client_key* to full capacity.

        Args:
            client_key: Client identifier whose bucket should be restored.
        """
        with self._lock:
            self._buckets[client_key] = _Bucket(tokens=self._capacity)

    def clear(self) -> None:
        """Remove all client buckets, effectively resetting every rate limit."""
        with self._lock:
            self._buckets.clear()

    @property
    def client_count(self) -> int:
        """Number of tracked client keys (including exhausted ones).

        Returns:
            Integer count of distinct client identifiers seen since creation
            or the last :meth:`clear` call.
        """
        with self._lock:
            return len(self._buckets)


def make_rate_limiter(
    capacity: float = 60.0,
    rate_per_second: float = 1.0,
    refill_rate: float | None = None,
) -> TokenBucketRateLimiter:
    """Factory returning a new :class:`TokenBucketRateLimiter` with defaults.

    *refill_rate* is an alias for *rate_per_second*; if both are given
    *refill_rate* takes precedence.  Pass ``refill_rate=0.0`` to create a
    bucket that never refills (useful for testing exhaustion scenarios).
    """
    effective_rate = refill_rate if refill_rate is not None else rate_per_second
    if effective_rate == 0.0:
        effective_rate = 1e-9  # effectively zero but passes validation
    return TokenBucketRateLimiter(capacity=capacity, rate_per_second=effective_rate)


__all__ = [
    "TokenBucketRateLimiter",
    "allow_burst",
    "bulk_allow",
    "burst_capacity_fraction",
    "client_stats",
    "is_rate_limited",
    "limiter_utilization",
    "make_rate_limiter",
    "make_strict_limiter",
    "prune_idle_clients",
    "reset_client",
    "reset_limiter",
]


def limiter_utilization(limiter: TokenBucketRateLimiter, client_key: str = "default") -> float:
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


def reset_limiter(limiter: TokenBucketRateLimiter, client_key: str = "default") -> None:
    """Reset a client's token bucket to full capacity.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: The client identifier. Default "default".
    """
    limiter.reset(client_key)


def is_rate_limited(limiter: TokenBucketRateLimiter, client_key: str = "default", cost: float = 1.0) -> bool:
    """Check whether a client's next request would be rate-limited WITHOUT consuming tokens.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: The client identifier. Default "default".
        cost: Token cost of the hypothetical request. Default 1.0.

    Returns:
        True if the next is_allowed call would be denied, else False.
    """
    return limiter.remaining_tokens(client_key) < cost


def make_strict_limiter(max_per_second: float) -> TokenBucketRateLimiter:
    """Create a rate limiter with capacity = max_per_second (burst of 1 second).

    Args:
        max_per_second: Allowed requests per second.

    Returns:
        TokenBucketRateLimiter configured for the given rate.
    """
    return make_rate_limiter(capacity=max_per_second, rate_per_second=max_per_second)


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


def prune_idle_clients(limiter: TokenBucketRateLimiter, max_idle_seconds: float) -> int:
    """Remove client buckets that have been idle longer than *max_idle_seconds*.

    A bucket is considered idle when ``time.monotonic() - bucket.last_refill``
    exceeds *max_idle_seconds*. Pruning keeps memory bounded in long-running
    processes with many ephemeral client keys.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.
        max_idle_seconds: Maximum time (in seconds) since last activity.

    Returns:
        Number of buckets removed.
    """
    now = time.monotonic()
    removed = 0
    with limiter._lock:
        idle_keys = [k for k, b in limiter._buckets.items() if now - b.last_refill > max_idle_seconds]
        for k in idle_keys:
            del limiter._buckets[k]
            removed += 1
    if removed:
        logger.debug("prune_idle_clients: removed %d idle buckets", removed)
    return removed


def client_stats(limiter: TokenBucketRateLimiter) -> dict[str, object]:
    """Return a snapshot of all tracked client keys and their token levels.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.

    Returns:
        Dict with 'client_count' and 'clients' (list of dicts with key, tokens).
    """
    with limiter._lock:
        snapshot = [{"key": k, "tokens": round(b.tokens, 4)} for k, b in limiter._buckets.items()]
    return {"client_count": len(snapshot), "clients": snapshot}


def allow_burst(limiter: TokenBucketRateLimiter, client_key: str, n: int) -> bool:
    """Attempt to consume *n* tokens at once (burst request).

    Returns True and deducts *n* tokens if the client has sufficient capacity;
    returns False without modifying the bucket otherwise.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.
        client_key: Unique client identifier.
        n: Number of tokens to consume in a single burst.

    Returns:
        True when the burst is permitted; False when insufficient tokens remain.

    Raises:
        ValueError: If *n* < 1.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    with limiter._lock:
        if client_key not in limiter._buckets:
            limiter._buckets[client_key] = _Bucket(tokens=limiter._capacity)
        bucket = limiter._buckets[client_key]
        limiter._refill(bucket)
        if bucket.tokens >= n:
            bucket.tokens -= n
            return True
        return False


def reset_client(limiter: TokenBucketRateLimiter, client_key: str) -> bool:
    """Reset a specific client's token bucket to full capacity.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.
        client_key: Identifier of the client to reset.

    Returns:
        True if the client existed and was reset; False if not found.
    """
    with limiter._lock:
        if client_key not in limiter._buckets:
            return False
        limiter._buckets[client_key] = _Bucket(tokens=limiter._capacity)
        return True


def active_client_count(limiter: TokenBucketRateLimiter) -> int:
    """Return the number of tracked client keys in the limiter.

    Args:
        limiter: A TokenBucketRateLimiter instance.

    Returns:
        Integer count of clients that have made at least one request.
    """
    return limiter.client_count


def total_consumed_tokens(limiter: TokenBucketRateLimiter, client_key: str | None = None) -> float:
    """Return total tokens consumed across all clients (or for one client).

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: If given, return consumed tokens for this client only.

    Returns:
        Non-negative float representing consumed token count.
    """
    capacity = limiter._capacity
    with limiter._lock:
        if client_key is not None:
            if client_key not in limiter._buckets:
                return 0.0
            bucket = limiter._buckets[client_key]
            limiter._refill(bucket)
            return round(max(0.0, capacity - bucket.tokens), 4)
        total = 0.0
        for bucket in limiter._buckets.values():
            limiter._refill(bucket)
            total += max(0.0, capacity - bucket.tokens)
        return round(total, 4)


def client_exists(limiter: TokenBucketRateLimiter, client_key: str) -> bool:
    """Return True if *client_key* has a tracked bucket in the limiter.

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: Client identifier to look up.

    Returns:
        True when the client has made at least one request, False otherwise.
    """
    with limiter._lock:
        return client_key in limiter._buckets


def bucket_fill_percentage(limiter: TokenBucketRateLimiter, client_key: str) -> float:
    """Return the fill percentage (0-100) of *client_key*'s token bucket.

    A new (unseen) client is considered full (100.0).

    Args:
        limiter: A TokenBucketRateLimiter instance.
        client_key: Client identifier.

    Returns:
        Float in [0.0, 100.0].
    """
    remaining = limiter.remaining_tokens(client_key)
    return round(min(100.0, remaining / limiter._capacity * 100.0), 4)


def bulk_allow(limiter: TokenBucketRateLimiter, client_keys: list[str]) -> dict[str, bool]:
    """Check rate-limit allowance for multiple clients in one call.

    Args:
        limiter: A :class:`TokenBucketRateLimiter` instance.
        client_keys: List of client identifiers to check.

    Returns:
        Dict mapping each client key to its allow/deny boolean.
    """
    return {key: limiter.is_allowed(key) for key in client_keys}
