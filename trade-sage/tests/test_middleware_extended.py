"""Extended tests for middleware classes."""

from __future__ import annotations

from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware, RequestTimingMiddleware


class TestRequestTimingMiddleware:
    def test_middleware_is_importable(self) -> None:
        assert RequestTimingMiddleware is not None

    def test_middleware_inherits_base_http(self) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware
        assert issubclass(RequestTimingMiddleware, BaseHTTPMiddleware)

    def test_has_dispatch_method(self) -> None:
        assert hasattr(RequestTimingMiddleware, "dispatch")


class TestRateLimitMiddleware:
    def test_middleware_is_importable(self) -> None:
        assert RateLimitMiddleware is not None

    def test_rate_limit_constant_positive(self) -> None:
        from app.middleware import RATE_LIMIT
        assert RATE_LIMIT > 0

    def test_rate_limit_value(self) -> None:
        from app.middleware import RATE_LIMIT
        assert RATE_LIMIT == 200


class TestCorrelationIDMiddleware:
    def test_middleware_is_importable(self) -> None:
        assert CorrelationIDMiddleware is not None

    def test_has_dispatch_method(self) -> None:
        assert hasattr(CorrelationIDMiddleware, "dispatch")
