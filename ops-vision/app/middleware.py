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

        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
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
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        bucket.append(now)
        response = await call_next(request)
        remaining = max(0, self.max_requests - len(bucket))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
)

MAX_REQUEST_BYTES: int = 1 * 1024 * 1024  # 1 MiB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds a configurable byte limit.

    Attributes:
        max_bytes: Maximum allowed Content-Length in bytes.
    """

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        """Initialise with the byte limit.

        Args:
            app: The ASGI application to wrap.
            max_bytes: Maximum allowed request body size.
        """
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        """Reject oversized requests before forwarding."""
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_bytes:
            logger.warning(
                "Request too large: %s bytes from %s",
                content_length,
                request.url.path,
            )
            return JSONResponse(
                {"detail": f"Request body exceeds {self.max_bytes} bytes"},
                status_code=413,
            )
        return await call_next(request)
