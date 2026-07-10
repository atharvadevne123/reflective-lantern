"""Rate limiting middleware — sliding-window counter per client IP."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import settings

_WINDOW_SECONDS = 60.0
_requests: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    """Resolve the rate-limit bucket key for a request.

    Args:
        request: Incoming HTTP request.

    Returns:
        The client IP (or ``unknown`` when the transport gives none).
    """
    if request.client is None:
        return "unknown"
    return request.client.host


async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    """Reject requests beyond ``settings.rate_limit_per_minute`` with HTTP 429.

    Uses an in-memory sliding window per client IP. For multi-replica
    deployments swap this for a shared store (e.g. Redis).
    """
    key = _client_key(request)
    now = time.monotonic()
    window = _requests[key]

    while window and now - window[0] > _WINDOW_SECONDS:
        window.popleft()

    if len(window) >= settings.rate_limit_per_minute:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": "60"},
        )

    window.append(now)
    return await call_next(request)


def reset_rate_limiter() -> None:
    """Clear all rate-limit windows (used by tests)."""
    _requests.clear()
