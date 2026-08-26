"""Tests for domain exceptions."""

import pytest

from app.exceptions import (
    BatchTooLargeError,
    FeatureMismatchError,
    ModelNotLoadedError,
    RetrieverNotReadyError,
    ThreatLensError,
)


@pytest.mark.parametrize(
    "exc",
    [
        ModelNotLoadedError,
        RetrieverNotReadyError,
    ],
)
def test_simple_errors_subclass_base(exc: type) -> None:
    assert issubclass(exc, ThreatLensError)
    assert str(exc()) != ""


def test_feature_mismatch_carries_counts() -> None:
    err = FeatureMismatchError(expected=28, received=25)
    assert err.expected == 28
    assert err.received == 25
    assert "28" in str(err) and "25" in str(err)


def test_batch_too_large_carries_sizes() -> None:
    err = BatchTooLargeError(size=500, maximum=100)
    assert err.size == 500
    assert err.maximum == 100
    assert "500" in str(err)


def test_all_errors_are_catchable_as_base() -> None:
    for err in (
        ModelNotLoadedError(),
        RetrieverNotReadyError(),
        FeatureMismatchError(1, 2),
        BatchTooLargeError(3, 2),
    ):
        with pytest.raises(ThreatLensError):
            raise err
