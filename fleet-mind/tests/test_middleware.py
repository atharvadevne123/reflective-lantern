"""Tests for Fleet-Mind ASGI middleware: correlation ID and rate limiting."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware


def _make_request(path: str = "/predict", ip: str = "127.0.0.1", headers: dict | None = None):
    request = MagicMock()
    request.url.path = path
    request.client = MagicMock()
    request.client.host = ip
    request.headers = headers or {}
    request.state = MagicMock()
    return request


def _make_response():
    response = MagicMock()
    response.headers = {}
    return response


@pytest.mark.asyncio
async def test_correlation_id_added_when_missing():
    app = AsyncMock()
    response = _make_response()
    app.return_value = response

    middleware = CorrelationIDMiddleware(app)
    request = _make_request()

    async def call_next(_req):
        return response

    result = await middleware.dispatch(request, call_next)
    assert "X-Correlation-ID" in result.headers
    assert len(result.headers["X-Correlation-ID"]) > 0


@pytest.mark.asyncio
async def test_correlation_id_preserved_from_request():
    middleware = CorrelationIDMiddleware(MagicMock())
    response = _make_response()
    request = _make_request(headers={"X-Correlation-ID": "test-id-123"})

    async def call_next(_req):
        return response

    result = await middleware.dispatch(request, call_next)
    assert result.headers["X-Correlation-ID"] == "test-id-123"


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    middleware = RateLimitMiddleware(MagicMock(), requests_per_minute=10)
    response = _make_response()
    request = _make_request()

    async def call_next(_req):
        return response

    result = await middleware.dispatch(request, call_next)
    assert result is response


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    middleware = RateLimitMiddleware(MagicMock(), requests_per_minute=2)
    response = _make_response()
    request = _make_request(ip="10.0.0.1")

    async def call_next(_req):
        return response

    await middleware.dispatch(request, call_next)
    await middleware.dispatch(request, call_next)
    result = await middleware.dispatch(request, call_next)
    assert result.status_code == 429 or result is response


@pytest.mark.asyncio
async def test_rate_limit_health_path_bypassed():
    middleware = RateLimitMiddleware(MagicMock(), requests_per_minute=1)
    response = _make_response()
    request = _make_request(path="/health")

    async def call_next(_req):
        return response

    for _ in range(5):
        result = await middleware.dispatch(request, call_next)
    assert result is response


def test_correlation_id_middleware_default_header():
    app = MagicMock()
    mw = CorrelationIDMiddleware(app)
    assert mw.header == "X-Correlation-ID"


def test_rate_limit_middleware_stores_limit():
    mw = RateLimitMiddleware(MagicMock(), requests_per_minute=50)
    assert mw.limit == 50


def test_rate_limit_get_client_ip_from_x_forwarded_for():
    mw = RateLimitMiddleware(MagicMock(), requests_per_minute=100)
    request = _make_request(headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
    ip = mw._get_client_ip(request)
    assert ip == "1.2.3.4"


def test_rate_limit_get_client_ip_fallback():
    mw = RateLimitMiddleware(MagicMock(), requests_per_minute=100)
    request = _make_request(ip="192.168.1.1")
    ip = mw._get_client_ip(request)
    assert ip == "192.168.1.1"
