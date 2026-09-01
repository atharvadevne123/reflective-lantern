"""Rate limiting middleware."""

import logging
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Fixed-window rate limiter keyed by client IP.

    Keeps an in-process deque of hit timestamps per client. This is per-worker
    state — behind multiple workers, use a shared store such as Redis instead.
    """

    def __init__(self, limit_per_minute: int = 120) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for `key` and report whether it is within the limit."""
        current = time.monotonic() if now is None else now
        window = self._hits[key]
        cutoff = current - WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(current)
        return True

    def remaining(self, key: str) -> int:
        """Return how many requests `key` may still make in the current window."""
        return max(0, self.limit - len(self._hits[key]))

    def reset(self) -> None:
        """Drop all recorded hits. Intended for tests."""
        self._hits.clear()


def build_rate_limit_middleware(limiter: RateLimiter) -> Any:
    """Return an ASGI middleware function enforcing `limiter`."""

    async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            logger.warning("Rate limit exceeded for %s", client)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(limiter.remaining(client))
        return response

    return rate_limit_middleware
