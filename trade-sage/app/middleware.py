"""Rate limiting and correlation ID middleware."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_request_counts: dict[str, list[float]] = defaultdict(list)
_lock = Lock()
RATE_LIMIT = 200  # requests per minute


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _lock:
            window = _request_counts[client_ip]
            _request_counts[client_ip] = [t for t in window if now - t < 60.0]
            if len(_request_counts[client_ip]) >= RATE_LIMIT:
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                )
            _request_counts[client_ip].append(now)
        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
