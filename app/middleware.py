"""Rate limiting middleware backed by an in-process sliding window."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from types import SimpleNamespace
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0

settings: Any = SimpleNamespace(rate_limit_per_minute=120)

# Module-level request store: {client_key: deque[timestamp]}
_requests: dict[str, deque[float]] = {}


def reset_rate_limiter() -> None:
    """Clear all recorded request windows (used in tests and on startup)."""
    _requests.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject clients exceeding `limit` requests per rolling minute.

    The window is per-process and in-memory, which is sufficient for a single
    replica. Multi-replica deployments should move this to Redis.
    """

    def __init__(self, app, limit: int = 120) -> None:
        super().__init__(app)
        self.limit = limit

    def _client_key(self, request: Request) -> str:
        """Return the originating IP, preferring X-Forwarded-For over direct client."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Enforce per-client request rate limits using a sliding window."""
        key = self._client_key(request)
        now = time.monotonic()

        if key not in _requests:
            _requests[key] = deque()
        bucket = _requests[key]

        current_limit = getattr(settings, "rate_limit_per_minute", self.limit)
        while bucket and now - bucket[0] > WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= current_limit:
            retry_after = int(WINDOW_SECONDS - (now - bucket[0])) + 1
            logger.warning("Rate limit exceeded for %s (%d hits)", key, len(bucket))
            return JSONResponse(
                status_code=429,
                content={"error": "RateLimitExceeded", "detail": "Too many requests"},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(current_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, current_limit - len(bucket)))
        response.headers["x-correlation-id"] = correlation_id
        return response
