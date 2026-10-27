"""Extended tests for threat-lens app/exceptions.py."""

from __future__ import annotations

import pytest

from app.exceptions import (
    BatchTooLargeError,
    FeatureMismatchError,
    ModelNotLoadedError,
    RetrieverNotReadyError,
    ThreatLensError,
)


class TestThreatLensError:
    def test_is_exception(self) -> None:
        assert issubclass(ThreatLensError, Exception)

    def test_can_raise(self) -> None:
        with pytest.raises(ThreatLensError):
            raise ThreatLensError("base error")


class TestModelNotLoadedError:
    def test_is_threat_lens_error(self) -> None:
        assert issubclass(ModelNotLoadedError, ThreatLensError)

    def test_default_message(self) -> None:
        err = ModelNotLoadedError()
        assert "not loaded" in str(err).lower()

    def test_custom_message(self) -> None:
        err = ModelNotLoadedError("no model file")
        assert "no model file" in str(err)


class TestFeatureMismatchError:
    def test_is_threat_lens_error(self) -> None:
        assert issubclass(FeatureMismatchError, ThreatLensError)

    def test_stores_expected_received(self) -> None:
        err = FeatureMismatchError(expected=10, received=8)
        assert err.expected == 10
        assert err.received == 8

    def test_message_contains_counts(self) -> None:
        err = FeatureMismatchError(expected=10, received=8)
        assert "10" in str(err)
        assert "8" in str(err)


class TestRetrieverNotReadyError:
    def test_is_threat_lens_error(self) -> None:
        assert issubclass(RetrieverNotReadyError, ThreatLensError)

    def test_default_message(self) -> None:
        err = RetrieverNotReadyError()
        assert "index" in str(err).lower() or "not built" in str(err).lower()


class TestBatchTooLargeError:
    def test_is_threat_lens_error(self) -> None:
        assert issubclass(BatchTooLargeError, ThreatLensError)

    def test_stores_size_maximum(self) -> None:
        err = BatchTooLargeError(size=200, maximum=100)
        assert err.size == 200
        assert err.maximum == 100

    def test_message_contains_values(self) -> None:
        err = BatchTooLargeError(size=200, maximum=100)
        assert "200" in str(err)
        assert "100" in str(err)
