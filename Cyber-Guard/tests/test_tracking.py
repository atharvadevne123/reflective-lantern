"""Tests for Cyber-Guard app/tracking.py."""

from __future__ import annotations

from unittest.mock import patch

from app.tracking import is_tracking_enabled, track_run


class TestIsTrackingEnabled:
    def test_false_without_mlflow(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", False):
            assert is_tracking_enabled() is False

    def test_false_without_uri(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", True), patch(
            "app.tracking.MLFLOW_TRACKING_URI", ""
        ):
            assert is_tracking_enabled() is False

    def test_true_with_both(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", True), patch(
            "app.tracking.MLFLOW_TRACKING_URI", "http://mlflow:5000"
        ):
            assert is_tracking_enabled() is True


class TestTrackRun:
    def test_noop_context_when_disabled(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", False):
            entered = False
            with track_run("test-run"):
                entered = True
            assert entered

    def test_noop_context_without_uri(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", True), patch(
            "app.tracking.MLFLOW_TRACKING_URI", ""
        ):
            side_effects = []
            with track_run("my-run"):
                side_effects.append("inside")
            assert "inside" in side_effects

    def test_context_manager_is_reentrant(self) -> None:
        with patch("app.tracking._MLFLOW_AVAILABLE", False):
            results = []
            with track_run("outer"):
                with track_run("inner"):
                    results.append("inner")
                results.append("outer")
            assert results == ["inner", "outer"]
