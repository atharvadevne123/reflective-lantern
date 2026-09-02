"""In-process sliding-window rate limiting middleware.

Suitable for a single API instance. Behind more than one replica, swap the
in-memory ``_HITS`` map for a shared Redis counter -- the interface here is
deliberately narrow so that substitution stays local to this module.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60

_HITS: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    """Derive a rate-limit bucket key for a request.

    Prefers the left-most ``X-Forwarded-For`` hop so the limit follows the
    real caller when the service sits behind a proxy.

    Args:
        request: The incoming request.

    Returns:
        A string key identifying the caller.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune(bucket: deque[float], now: float) -> None:
    """Drop timestamps that have fallen out of the sliding window.

    Args:
        bucket: Timestamps of recent hits, oldest first.
        now: Current monotonic-ish timestamp.
    """
    cutoff = now - WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.popleft()


def reset_rate_limit_state() -> None:
    """Clear all rate-limit buckets. Intended for tests."""
    _HITS.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject callers exceeding ``limit_per_minute`` requests in a 60s window."""

    def __init__(self, app, limit_per_minute: int = 120) -> None:
        super().__init__(app)
        self.limit = limit_per_minute

    async def dispatch(self, request: Request, call_next):
        """Count the request and either pass it through or return HTTP 429.

        Args:
            request: The incoming request.
            call_next: The downstream handler.

        Returns:
            The downstream response, or a 429 JSON response when over limit.
        """
        # Health checks must never be rate limited: an orchestrator probing
        # liveness would otherwise be able to take the service out itself.
        if request.url.path.endswith("/health"):
            return await call_next(request)

        key = _client_key(request)
        now = time.time()
        bucket = _HITS[key]
        _prune(bucket, now)

        if len(bucket) >= self.limit:
            retry_after = int(WINDOW_SECONDS - (now - bucket[0])) + 1
            logger.warning("rate limit exceeded key=%s hits=%d", key, len(bucket))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "rate limit exceeded",
                    "limit_per_minute": self.limit,
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.limit - len(bucket)))
        return response
