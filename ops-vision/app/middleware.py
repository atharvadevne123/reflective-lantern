"""Correlation-ID injection and rate-limiting middleware for Ops-Vision."""

import logging
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Correlation-ID header into every request and response.

    If the incoming request already carries an X-Correlation-ID header its
    value is forwarded; otherwise a new UUID-4 is generated.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request and attach a correlation ID."""
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        logger.info(
            "Request %s %s correlation_id=%s",
            request.method,
            request.url.path,
            correlation_id,
        )

        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter keyed by client IP address.

    Allows at most `max_requests` requests per `window_seconds` per IP.
    Returns HTTP 429 when the limit is exceeded.

    Attributes:
        max_requests: Maximum allowed requests per window.
        window_seconds: Rolling window length in seconds.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        """Initialise the rate limiter.

        Args:
            app: The ASGI application to wrap.
            max_requests: Requests allowed per IP per window.
            window_seconds: Duration of the rolling window.
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, honouring X-Forwarded-For if present."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Enforce rate limit before forwarding the request."""
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()
        bucket = self._buckets[client_ip]

        while bucket and bucket[0] < now - self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": str(self.window_seconds)},
            )

        bucket.append(now)
        return await call_next(request)
