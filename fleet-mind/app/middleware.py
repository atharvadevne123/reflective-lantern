"""ASGI middleware: correlation IDs and simple in-process rate limiting."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique correlation ID to every request and response."""

    def __init__(self, app: ASGIApp, header: str = "X-Correlation-ID") -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get(self.header) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[self.header] = correlation_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter keyed by client IP (in-process, non-distributed)."""

    def __init__(self, app: ASGIApp, requests_per_minute: int = 100) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()
        window = self._buckets[client_ip]
        self._buckets[client_ip] = [t for t in window if now - t < 60.0]

        if len(self._buckets[client_ip]) >= self.limit:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Retry after 60 seconds."},
                headers={"Retry-After": "60"},
            )

        self._buckets[client_ip].append(now)
        return await call_next(request)
