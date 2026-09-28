"""Tests for domain exceptions."""
from __future__ import annotations

import pytest

from app.exceptions import (
    FeatureExtractionError,
    LogisticsFlowError,
    ModelNotLoadedError,
    RateLimitExceededError,
)


@pytest.mark.parametrize(
    ("exc_cls", "status"),
    [
        (ModelNotLoadedError, 503),
        (FeatureExtractionError, 422),
        (RateLimitExceededError, 429),
    ],
)
def test_status_codes(exc_cls, status) -> None:
    assert exc_cls().status_code == status


def test_custom_detail_overrides_default() -> None:
    exc = ModelNotLoadedError("model file missing")
    assert exc.detail == "model file missing"


def test_default_detail_used_when_omitted() -> None:
    assert ModelNotLoadedError().detail == "Model not loaded"


def test_all_inherit_base() -> None:
    for cls in (ModelNotLoadedError, FeatureExtractionError, RateLimitExceededError):
        assert issubclass(cls, LogisticsFlowError)
