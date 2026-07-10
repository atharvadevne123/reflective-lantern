"""Tests for custom exception hierarchy."""
from __future__ import annotations

import pytest


def test_cyber_sentinel_error_is_base() -> None:
    from app.exceptions import CyberSentinelError, ModelNotTrainedError

    assert issubclass(ModelNotTrainedError, CyberSentinelError)


def test_model_not_trained_error() -> None:
    from app.exceptions import ModelNotTrainedError

    with pytest.raises(ModelNotTrainedError):
        raise ModelNotTrainedError("model not ready")


def test_feature_extraction_error() -> None:
    from app.exceptions import FeatureExtractionError

    err = FeatureExtractionError("bad input")
    assert str(err) == "bad input"


def test_drift_detection_error() -> None:
    from app.exceptions import DriftDetectionError

    assert issubclass(DriftDetectionError, Exception)


def test_database_error() -> None:
    from app.exceptions import DatabaseError

    assert issubclass(DatabaseError, Exception)


def test_all_exceptions_catchable_as_base() -> None:
    from app.exceptions import (
        CyberSentinelError,
        DatabaseError,
        DriftDetectionError,
        FeatureExtractionError,
        ModelNotTrainedError,
    )

    for exc_cls in [ModelNotTrainedError, FeatureExtractionError, DriftDetectionError, DatabaseError]:
        try:
            raise exc_cls("test")
        except CyberSentinelError:
            pass
