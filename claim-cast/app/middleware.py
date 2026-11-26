"""Correlation ID logging and rate limiting middleware."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

_request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 60
RATE_WINDOW = 60.0


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("correlation_id")
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_WINDOW
        timestamps = [t for t in _request_counts[client_ip] if t > window_start]
        timestamps.append(now)
        _request_counts[client_ip] = timestamps
        if len(timestamps) > RATE_LIMIT:
            logger.warning("rate_limit_exceeded", client_ip=client_ip, count=len(timestamps))
            return JSONResponse(
                {"detail": "Rate limit exceeded. Max 60 requests/minute."},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
