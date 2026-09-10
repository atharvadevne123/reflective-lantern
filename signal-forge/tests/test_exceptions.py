"""Tests for custom exception hierarchy."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.exceptions import (
    SignalForgeError,
    FeatureExtractionError,
    ModelNotFoundError,
    PredictionError,
    DriftDetectionError,
    DatabaseError,
    ValidationError,
)


def test_signal_forge_error_base():
    with pytest.raises(SignalForgeError):
        raise SignalForgeError("base error")


@pytest.mark.parametrize("exc_class,msg", [
    (FeatureExtractionError, "feature failed"),
    (ModelNotFoundError, "model not found"),
    (PredictionError, "prediction failed"),
    (DriftDetectionError, "drift error"),
    (DatabaseError, "db error"),
    (ValidationError, "validation failed"),
])
def test_subclasses_inherit_base(exc_class, msg):
    err = exc_class(msg)
    assert isinstance(err, SignalForgeError)
    assert str(err) == msg


def test_catch_as_base():
    try:
        raise FeatureExtractionError("caught")
    except SignalForgeError as e:
        assert "caught" in str(e)
