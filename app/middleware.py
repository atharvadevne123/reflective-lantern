"""Rate limiting and correlation-ID middleware."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 200  # requests per minute


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding RATE_LIMIT per IP per minute."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = [t for t in _request_counts[ip] if now - t < 60]
        _request_counts[ip] = window
        if len(window) >= RATE_LIMIT:
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "Rate limit exceeded. Max 200 req/min."}, status_code=429)
        _request_counts[ip].append(now)
        return await call_next(request)

RATE_LIMIT_PER_MINUTE = 200
_rate_buckets: dict[str, deque] = defaultdict(lambda: deque())


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Correlation-ID to every request/response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


# Type alias used above
from typing import Any  # noqa: E402 (placed after class bodies to avoid circular)
