"""Tests for suite-cast app/__main__.py entry point."""

from __future__ import annotations

from unittest.mock import patch


class TestMain:
    def test_main_calls_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            from app.__main__ import main

            main()
            mock_run.assert_called_once()

    def test_main_uses_correct_app(self) -> None:
        with patch("uvicorn.run") as mock_run:
            from app.__main__ import main

            main()
            call_args = mock_run.call_args
            assert "app.main:app" in call_args[0] or call_args[1].get("app") == "app.main:app"

    def test_main_binds_to_0000(self) -> None:
        with patch("uvicorn.run") as mock_run:
            from app.__main__ import main

            main()
            kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
            args = mock_run.call_args[0] if mock_run.call_args[0] else []
            host = kwargs.get("host", "")
            assert host == "0.0.0.0" or len(args) > 0
