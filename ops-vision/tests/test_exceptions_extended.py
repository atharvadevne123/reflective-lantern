"""Extended tests for ops-vision app/exceptions.py."""

from __future__ import annotations

import pytest

from app.exceptions import (
    DriftMonitorError,
    FeatureEngineeringError,
    ModelNotLoadedError,
    OpsVisionError,
    RunbookIndexError,
)


class TestOpsVisionError:
    def test_is_exception(self) -> None:
        assert issubclass(OpsVisionError, Exception)

    def test_can_raise(self) -> None:
        with pytest.raises(OpsVisionError):
            raise OpsVisionError("test")


class TestModelNotLoadedError:
    def test_is_ops_vision_error(self) -> None:
        assert issubclass(ModelNotLoadedError, OpsVisionError)

    def test_message_contains_model(self) -> None:
        err = ModelNotLoadedError()
        assert "model" in str(err).lower()

    def test_can_catch_as_base(self) -> None:
        with pytest.raises(OpsVisionError):
            raise ModelNotLoadedError()


class TestFeatureEngineeringError:
    def test_is_ops_vision_error(self) -> None:
        assert issubclass(FeatureEngineeringError, OpsVisionError)

    def test_default_message(self) -> None:
        err = FeatureEngineeringError()
        assert "feature" in str(err).lower()

    def test_custom_detail(self) -> None:
        err = FeatureEngineeringError("missing column X")
        assert "missing column X" in str(err)
        assert err.detail == "missing column X"


class TestDriftMonitorError:
    def test_is_ops_vision_error(self) -> None:
        assert issubclass(DriftMonitorError, OpsVisionError)

    def test_reason_in_message(self) -> None:
        err = DriftMonitorError("no reference data")
        assert "no reference data" in str(err)
        assert err.reason == "no reference data"


class TestRunbookIndexError:
    def test_is_ops_vision_error(self) -> None:
        assert issubclass(RunbookIndexError, OpsVisionError)

    def test_reason_in_message(self) -> None:
        err = RunbookIndexError("index corrupt")
        assert "index corrupt" in str(err)
        assert err.reason == "index corrupt"


class TestOpsVisionErrorRepr:
    def test_repr_includes_class_name(self) -> None:
        err = OpsVisionError("test msg")
        assert "OpsVisionError" in repr(err)

    def test_repr_includes_message(self) -> None:
        err = OpsVisionError("test message")
        assert "test message" in repr(err)

    def test_subclass_repr_uses_subclass_name(self) -> None:
        err = ModelNotLoadedError()
        assert "ModelNotLoadedError" in repr(err)


class TestFeatureEngineeringErrorExtra:
    def test_default_detail_message(self) -> None:
        err = FeatureEngineeringError()
        assert "Feature engineering failed" in str(err)

    def test_custom_detail_stored(self) -> None:
        err = FeatureEngineeringError("pipeline transform error")
        assert err.detail == "pipeline transform error"

    def test_inherits_from_ops_vision_error(self) -> None:
        assert issubclass(FeatureEngineeringError, OpsVisionError)
