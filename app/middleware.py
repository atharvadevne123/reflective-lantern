"""Custom ASGI middleware for Realty-Edge.

Provides:
- CorrelationIDMiddleware: stamps every request/response with X-Correlation-ID
- RequestLoggingMiddleware: logs method, path, status, and latency at INFO level
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Read or generate X-Correlation-ID and expose it on request.state."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Read or generate X-Correlation-ID, attach it to request state, and echo in response."""
        corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = corr_id
        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
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
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
