"""Extended exception hierarchy tests — third pass."""
from __future__ import annotations

import pytest

from app.exceptions import (
    DriftMonitorError,
    FeatureEngineeringError,
    ModelNotLoadedError,
    OpsVisionError,
    RunbookIndexError,
)


def test_ops_vision_error_is_exception() -> None:
    assert issubclass(OpsVisionError, Exception)


def test_model_not_loaded_is_ops_vision_error() -> None:
    assert issubclass(ModelNotLoadedError, OpsVisionError)


def test_feature_engineering_is_ops_vision_error() -> None:
    assert issubclass(FeatureEngineeringError, OpsVisionError)


def test_drift_monitor_is_ops_vision_error() -> None:
    assert issubclass(DriftMonitorError, OpsVisionError)


def test_runbook_index_is_ops_vision_error() -> None:
    assert issubclass(RunbookIndexError, OpsVisionError)


def test_model_not_loaded_raises_as_ops_vision() -> None:
    with pytest.raises(OpsVisionError):
        raise ModelNotLoadedError()


def test_model_not_loaded_message_mentions_load_artifacts() -> None:
    exc = ModelNotLoadedError()
    assert "_load_artifacts" in str(exc)


def test_feature_engineering_default_message() -> None:
    exc = FeatureEngineeringError()
    assert "failed" in str(exc).lower()


def test_feature_engineering_custom_detail_stored() -> None:
    exc = FeatureEngineeringError("bad pipeline step")
    assert exc.detail == "bad pipeline step"
    assert "bad pipeline step" in str(exc)


def test_drift_monitor_reason_stored() -> None:
    exc = DriftMonitorError("too few samples")
    assert exc.reason == "too few samples"
    assert "too few samples" in str(exc)


def test_runbook_index_reason_stored() -> None:
    exc = RunbookIndexError("FAISS build failed")
    assert exc.reason == "FAISS build failed"


@pytest.mark.parametrize("reason", ["timeout", "empty index", "dimension mismatch"])
def test_runbook_index_various_reasons(reason: str) -> None:
    exc = RunbookIndexError(reason)
    assert reason in str(exc)


def test_ops_vision_repr_includes_class_name() -> None:
    exc = OpsVisionError("test msg")
    assert "OpsVisionError" in repr(exc)


def test_feature_engineering_repr_includes_class() -> None:
    exc = FeatureEngineeringError("bad")
    assert "FeatureEngineeringError" in repr(exc)


@pytest.mark.parametrize("detail", ["empty df", "NaN in features", "wrong shape"])
def test_feature_engineering_various_details(detail: str) -> None:
    exc = FeatureEngineeringError(detail)
    assert exc.detail == detail


def test_drift_monitor_raises_as_ops_vision() -> None:
    with pytest.raises(OpsVisionError):
        raise DriftMonitorError("no reference window")
