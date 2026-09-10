"""FastAPI middleware: correlation ID logging and rate limiting."""

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Simple in-process token-bucket rate limiter
_request_timestamps: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60.0


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique correlation ID to every request and log it."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "method=%s path=%s status=%d duration_ms=%s corr_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            correlation_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds per IP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        timestamps = _request_timestamps[client_ip]
        _request_timestamps[client_ip] = [t for t in timestamps if t > window_start]
        _request_timestamps[client_ip].append(now)

        if len(_request_timestamps[client_ip]) > RATE_LIMIT_REQUESTS:
            logger.warning("Rate limit exceeded for IP: %s", client_ip)
            return Response(
                content='{"detail":"Rate limit exceeded. Max 60 requests per minute."}',
                status_code=429,
                media_type="application/json",
            )
        return await call_next(request)
