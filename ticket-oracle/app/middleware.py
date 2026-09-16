"""Custom FastAPI middleware for Ticket-Oracle.

Provides correlation ID injection and structured request logging
to make distributed tracing and log correlation straightforward.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject or propagate a correlation ID on every request/response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers[CORRELATION_ID_HEADER] = correlation_id

        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
                "correlation_id": correlation_id,
            },
        )
        return response


class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """Attach rate-limit metadata headers when slowapi limits are active."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if response.status_code == 429:
            response.headers["Retry-After"] = "60"
        return response
