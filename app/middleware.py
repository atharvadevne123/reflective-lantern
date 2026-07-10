"""FastAPI middleware: correlation ID and rate limiting."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE = 200
_rate_buckets: dict[str, deque] = defaultdict(lambda: deque())


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID into each request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-process per-IP rate limiter (RATE_LIMIT_PER_MINUTE req/min)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _rate_buckets[client_ip]

        # evict timestamps older than 60 seconds
        while bucket and now - bucket[0] > 60.0:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            logger.warning("rate limit exceeded for %s", client_ip)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in 60 seconds."},
                headers={"Retry-After": "60"},
            )

        bucket.append(now)
        return await call_next(request)
