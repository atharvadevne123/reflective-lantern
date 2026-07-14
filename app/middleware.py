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
    """Clear all rate-limit tracking state (useful in tests)."""
    _request_counts.clear()
    _rate_buckets.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding settings.rate_limit_per_minute per IP per minute."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
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
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
