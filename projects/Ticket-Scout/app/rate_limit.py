"""Token-bucket rate limiting middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_RATE_LIMIT_PER_MINUTE = 60
_WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-process token-bucket rate limiter keyed by client IP."""

    def __init__(self, app, requests_per_minute: int = _RATE_LIMIT_PER_MINUTE) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._counts: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        ip = self._client_ip(request)
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS

        timestamps = [t for t in self._counts[ip] if t > window_start]
        self._counts[ip] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for IP %s", ip)
            return Response(
                content='{"detail":"Rate limit exceeded. Max 60 requests/minute."}',
                status_code=429,
                media_type="application/json",
            )

        self._counts[ip].append(now)
        return await call_next(request)
