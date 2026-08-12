"""
Custom middleware for the Price-Prophet FastAPI application.

The :class:`TimingMiddleware` measures end-to-end request latency and
injects it into the response as an ``X-Response-Time`` header (in ms)
so that downstream clients and proxies can observe API performance
without needing access to server-side logs.
"""

from __future__ import annotations

import time
import logging

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that appends request-duration headers.

    Adds the following header to every response:

    ``X-Response-Time``
        Wall-clock duration of the request in milliseconds, formatted
        as a string with two decimal places (e.g. ``"12.34"``).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request and record elapsed time.

        Parameters
        ----------
        request:
            Incoming ASGI request.
        call_next:
            Next middleware or route handler in the stack.

        Returns
        -------
        Response
            Original response with the ``X-Response-Time`` header added.
        """
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1_000.0
        logger.debug(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}"
        return response


def add_middleware(app: FastAPI) -> None:
    """Attach :class:`TimingMiddleware` to *app*.

    This is a convenience wrapper so application code does not need to
    import :class:`TimingMiddleware` directly.

    Parameters
    ----------
    app:
        The FastAPI application instance to instrument.
    """
    app.add_middleware(TimingMiddleware)
