"""FastAPI middleware for correlation IDs and rate limiting."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# In-memory rate limit store: {ip: [timestamps]}
_request_log: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60.0


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Correlation-ID header to every request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to request state and response headers.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response with X-Correlation-ID header.
        """
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        logger.info(
            "Request: %s %s correlation_id=%s",
            request.method,
            request.url.path,
            correlation_id,
        )

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter: 100 requests per 60 seconds per IP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce rate limit for the requesting IP address.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response, or 429 if rate limit exceeded.
        """
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Evict timestamps outside the window
        _request_log[client_ip] = [ts for ts in _request_log[client_ip] if now - ts < RATE_LIMIT_WINDOW]

        if len(_request_log[client_ip]) >= RATE_LIMIT_REQUESTS:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        _request_log[client_ip].append(now)
        return await call_next(request)
