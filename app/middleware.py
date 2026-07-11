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

CORRELATION_ID_HEADER = "X-Correlation-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-Ms"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Correlation-ID to every request/response."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Read or generate X-Correlation-ID, attach it to request state, and echo in response."""
        corr_id = request.headers.get(CORRELATION_ID_HEADER, str(uuid.uuid4()))
        request.state.correlation_id = corr_id
        response: Response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = corr_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log HTTP method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Process the request and emit a structured INFO log with timing."""
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        corr_id = getattr(request.state, "correlation_id", "-")
        logger.info(
            "%s %s %d %.1fms corr=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            corr_id,
        )
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.1f}"
        return response
