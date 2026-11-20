"""Extended tests for ops-vision app/exceptions.py custom exception hierarchy."""

from __future__ import annotations

import pytest

from app.exceptions import (
    DriftMonitorError,
    FeatureEngineeringError,
    ModelNotLoadedError,
    OpsVisionError,
    RunbookIndexError,
)


class TestExceptionHierarchy:
    def test_model_not_loaded_is_ops_vision_error(self) -> None:
        assert issubclass(ModelNotLoadedError, OpsVisionError)

    def test_feature_engineering_error_is_ops_vision_error(self) -> None:
        assert issubclass(FeatureEngineeringError, OpsVisionError)

    def test_drift_monitor_error_is_ops_vision_error(self) -> None:
        assert issubclass(DriftMonitorError, OpsVisionError)

    def test_runbook_index_error_is_ops_vision_error(self) -> None:
        assert issubclass(RunbookIndexError, OpsVisionError)

    def test_ops_vision_error_is_exception(self) -> None:
        assert issubclass(OpsVisionError, Exception)


class TestExceptionRaisable:
    def test_raise_model_not_loaded(self) -> None:
        with pytest.raises(ModelNotLoadedError):
            raise ModelNotLoadedError("model missing")

    def test_raise_feature_engineering_error(self) -> None:
        with pytest.raises(FeatureEngineeringError):
            raise FeatureEngineeringError("bad features")

    def test_raise_drift_monitor_error(self) -> None:
        with pytest.raises(DriftMonitorError):
            raise DriftMonitorError("drift error")

    def test_raise_runbook_index_error(self) -> None:
        with pytest.raises(RunbookIndexError):
            raise RunbookIndexError("index broken")

    def test_caught_as_ops_vision_error(self) -> None:
        with pytest.raises(OpsVisionError):
            raise ModelNotLoadedError("model missing")


class TestExceptionMessages:
    def test_message_preserved(self) -> None:
        err = ModelNotLoadedError("test message")
        assert "test message" in str(err)

    def test_feature_error_has_message(self) -> None:
        err = FeatureEngineeringError("field missing")
        assert "field missing" in str(err)

    @pytest.mark.parametrize("exc_cls,msg", [
        (ModelNotLoadedError, "no model"),
        (FeatureEngineeringError, "bad input"),
        (DriftMonitorError, "drift"),
        (RunbookIndexError, "index error"),
    ])
    def test_all_exceptions_preserve_message(self, exc_cls, msg) -> None:
        err = exc_cls(msg)
        assert msg in str(err)
