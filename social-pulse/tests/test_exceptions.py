"""Tests for custom exception classes."""

from __future__ import annotations

import pytest

from app.exceptions import (
    DatabaseError,
    DriftCheckError,
    FeatureExtractionError,
    ModelNotLoadedError,
    SocialPulseError,
)


class TestSocialPulseError:
    def test_is_exception(self) -> None:
        assert issubclass(SocialPulseError, Exception)

    def test_message_stored(self) -> None:
        err = SocialPulseError("test message")
        assert err.message == "test message"

    def test_default_code(self) -> None:
        err = SocialPulseError("msg")
        assert err.code == "INTERNAL_ERROR"

    def test_custom_code(self) -> None:
        err = SocialPulseError("msg", code="CUSTOM")
        assert err.code == "CUSTOM"

    def test_str_representation(self) -> None:
        err = SocialPulseError("hello world")
        assert "hello world" in str(err)


class TestModelNotLoadedError:
    def test_is_social_pulse_error(self) -> None:
        err = ModelNotLoadedError()
        assert isinstance(err, SocialPulseError)

    def test_code(self) -> None:
        assert ModelNotLoadedError().code == "MODEL_NOT_LOADED"

    def test_message_contains_hint(self) -> None:
        assert "/train" in ModelNotLoadedError().message


class TestFeatureExtractionError:
    def test_code(self) -> None:
        assert FeatureExtractionError("bad input").code == "FEATURE_EXTRACTION_ERROR"

    def test_detail_in_message(self) -> None:
        err = FeatureExtractionError("missing column")
        assert "missing column" in err.message


class TestDatabaseError:
    def test_code(self) -> None:
        assert DatabaseError("timeout").code == "DATABASE_ERROR"

    def test_detail_in_message(self) -> None:
        err = DatabaseError("connection refused")
        assert "connection refused" in err.message


class TestDriftCheckError:
    def test_code(self) -> None:
        assert DriftCheckError("not enough data").code == "DRIFT_CHECK_ERROR"

    def test_detail_in_message(self) -> None:
        err = DriftCheckError("insufficient samples")
        assert "insufficient samples" in err.message


@pytest.mark.parametrize(
    "exc_cls,code",
    [
        (ModelNotLoadedError, "MODEL_NOT_LOADED"),
        (lambda: FeatureExtractionError("x"), "FEATURE_EXTRACTION_ERROR"),
        (lambda: DatabaseError("x"), "DATABASE_ERROR"),
        (lambda: DriftCheckError("x"), "DRIFT_CHECK_ERROR"),
    ],
)
def test_exception_codes(exc_cls, code: str) -> None:
    exc = exc_cls()
    assert exc.code == code
