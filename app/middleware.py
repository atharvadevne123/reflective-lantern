"""Rate limiting and correlation-ID middleware."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

settings = SimpleNamespace(rate_limit_per_minute=200)

_request_counts: dict[str, list[float]] = defaultdict(list)
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_requests = _request_counts


def reset_rate_limiter() -> None:
    """Clear all rate-limit tracking state (useful in tests).

    Resets both the sliding-window request counts and the token-bucket state
    so that each test starts with a clean slate.
    """
    _request_counts.clear()
    _rate_buckets.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding settings.rate_limit_per_minute per IP per minute."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Apply per-IP sliding-window rate limiting.

        Args:
            request: Incoming HTTP request.
            call_next: ASGI callable to forward compliant requests.

        Returns:
            The downstream response, or a 429 JSON response when the client
            has exceeded the configured rate limit.
        """
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        limit = settings.rate_limit_per_minute
        window = [t for t in _request_counts[ip] if now - t < 60]
        _request_counts[ip] = window
        if len(window) >= limit:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": f"Rate limit exceeded. Max {limit} req/min."},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        _request_counts[ip].append(now)
        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Correlation-ID to every request/response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Echo or generate an X-Correlation-ID header.

        Reads ``X-Correlation-ID`` from the incoming headers if present;
        otherwise generates a new UUID4 string. The chosen ID is stored on
        ``request.state.correlation_id`` and echoed back in the response
        headers.

        Args:
            request: Incoming HTTP request.
            call_next: ASGI callable to forward the request downstream.

        Returns:
            The downstream response with the correlation-id header attached.
        """
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


__all__ = [
    "CorrelationIDMiddleware",
    "RateLimitMiddleware",
    "reset_rate_limiter",
    "settings",
]
