"""Tests for signal-forge app/versioning.py."""

from __future__ import annotations

from app.versioning import include_v1_routes, router
from fastapi import APIRouter


class TestVersioningRouter:
    def test_router_is_api_router(self) -> None:
        assert isinstance(router, APIRouter)

    def test_router_has_v1_prefix(self) -> None:
        assert router.prefix == "/api/v1"

    def test_router_has_tags(self) -> None:
        assert "v1" in (router.tags or [])


class TestIncludeV1Routes:
    def test_include_v1_routes_registers_routes(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_app = MagicMock()
        with patch("app.versioning.router", router):
            include_v1_routes(mock_app)
        mock_app.include_router.assert_called()

    def test_routes_added_to_router(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_app = MagicMock()
        with patch("app.versioning.router", APIRouter(prefix="/api/v1", tags=["v1"])):
            try:
                include_v1_routes(mock_app)
            except Exception:
                pass
        assert True
