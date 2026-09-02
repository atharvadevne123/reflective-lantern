"""Rate limiting and request middleware for Temporal-Pulse."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-client-IP rate limiter.

    Keeps a deque of request timestamps per client and rejects requests
    exceeding RATE_LIMIT_REQUESTS within RATE_LIMIT_WINDOW_SECONDS.
    """

    def __init__(
        self, app, requests_per_window: int | None = None, window_seconds: int | None = None
    ) -> None:
        super().__init__(app)
        self.requests_per_window = requests_per_window or RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or RATE_LIMIT_WINDOW_SECONDS
        self._request_log: dict[str, deque] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        key = self._client_key(request)
        now = time.monotonic()
        window = self._request_log[key]

        # Evict timestamps outside the sliding window
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.requests_per_window:
            retry_after = int(window[0] + self.window_seconds - now) + 1
            logger.warning("Rate limit exceeded for client %s", key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        return await call_next(request)
