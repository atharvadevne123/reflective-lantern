"""Application-level middleware for Social-Pulse."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach and propagate a unique X-Request-ID header on every request.

    If the incoming request already carries an X-Request-ID header, that
    value is preserved; otherwise a new UUID4 is generated and attached to
    both the request state and the response headers.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request: attach correlation ID, time the call, log summary.

        Args:
            request: The incoming HTTP request.
            call_next: ASGI callable to forward the request to the next handler.

        Returns:
            The HTTP response with X-Request-ID and X-Response-Time-Ms headers.
        """
        request_id: str = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "method=%s path=%s status=%d latency_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add minimal security response headers to every HTTP response.

    Sets X-Content-Type-Options, X-Frame-Options, and
    Referrer-Policy on every outbound response.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Attach security headers to the response.

        Args:
            request: The incoming HTTP request.
            call_next: ASGI callable for the next handler.

        Returns:
            Response with security headers added.
        """
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
