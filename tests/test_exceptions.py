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
def test_status_codes(exc_cls, status):
    assert exc_cls().status_code == status


def test_custom_detail_overrides_default():
    exc = ModelNotLoadedError("model file missing")
    assert exc.detail == "model file missing"


def test_default_detail_used_when_omitted():
    assert ModelNotLoadedError().detail == "Model not loaded"


def test_all_inherit_base():
    for cls in (ModelNotLoadedError, FeatureExtractionError, RateLimitExceededError):
        assert issubclass(cls, LogisticsFlowError)


def test_exception_message_equals_detail():
    exc = ModelNotLoadedError("custom reason")
    assert str(exc) == "custom reason"


def test_logistics_flow_error_base_defaults():
    exc = LogisticsFlowError()
    assert exc.status_code == 500
    assert exc.detail == "Internal error"


def test_feature_extraction_default_detail():
    assert FeatureExtractionError().detail == "Feature extraction failed"


def test_rate_limit_default_detail():
    assert RateLimitExceededError().detail == "Rate limit exceeded"


def test_exception_is_exception_subclass():
    for cls in (LogisticsFlowError, ModelNotLoadedError, FeatureExtractionError, RateLimitExceededError):
        assert issubclass(cls, Exception)


@pytest.mark.parametrize(
    "exc_cls",
    [ModelNotLoadedError, FeatureExtractionError, RateLimitExceededError],
)
def test_custom_detail_preserved_in_message(exc_cls):
    exc = exc_cls("specific detail")
    assert "specific detail" in str(exc)


def test_register_exception_handlers_is_callable():
    from app.exceptions import register_exception_handlers
    from fastapi import FastAPI

    test_app = FastAPI()
    register_exception_handlers(test_app)  # should not raise
