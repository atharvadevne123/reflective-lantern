"""Rate limiting middleware backed by an in-process sliding window."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject clients exceeding `limit` requests per rolling minute.

    The window is per-process and in-memory, which is sufficient for a single
    replica. Multi-replica deployments should move this to Redis.
    """

    def __init__(self, app, limit: int = 120) -> None:
        super().__init__(app)
        self.limit = limit
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        key = self._client_key(request)
        now = time.monotonic()
        bucket = self._hits[key]

        while bucket and now - bucket[0] > WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= self.limit:
            retry_after = int(WINDOW_SECONDS - (now - bucket[0])) + 1
            logger.warning("Rate limit exceeded for %s (%d hits)", key, len(bucket))
            return JSONResponse(
                status_code=429,
                content={"error": "RateLimitExceeded", "detail": "Too many requests"},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.limit - len(bucket)))
        return response
