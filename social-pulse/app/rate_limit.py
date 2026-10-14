"""Token-bucket rate limiting middleware for Social-Pulse."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_DEFAULT_RATE: int = 60   # requests
_DEFAULT_WINDOW: int = 60  # seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Allows up to *rate* requests per *window* seconds per unique IP.
    Exempt paths (e.g. /health) bypass the limiter.
    """

    EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/version", "/docs", "/openapi.json"})

    def __init__(
        self,
        app: Any,
        rate: int = _DEFAULT_RATE,
        window: int = _DEFAULT_WINDOW,
    ) -> None:
        super().__init__(app)
        self.rate = rate
        self.window = window
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract the real client IP, honoring X-Forwarded-For."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Check rate limit and forward or reject the request."""
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()
        window_start = now - self.window

        # Prune old timestamps
        timestamps = self._buckets[client_ip]
        self._buckets[client_ip] = [t for t in timestamps if t > window_start]

        if len(self._buckets[client_ip]) >= self.rate:
            logger.warning("Rate limit exceeded for %s (%d reqs)", client_ip, len(self._buckets[client_ip]))
            return Response(
                content='{"detail":"Rate limit exceeded","code":"RATE_LIMITED"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.window),
                    "X-RateLimit-Limit": str(self.rate),
                    "X-RateLimit-Remaining": "0",
                },
            )

        self._buckets[client_ip].append(now)
        response: Response = await call_next(request)
        remaining = max(0, self.rate - len(self._buckets[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
