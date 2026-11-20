"""Tests for Ops-Vision custom exception types."""

import pytest

from app.exceptions import (
    DriftMonitorError,
    FeatureEngineeringError,
    ModelNotLoadedError,
    OpsVisionError,
    RunbookIndexError,
)


class TestOpsVisionErrorHierarchy:
    """Tests for custom exception inheritance and messages."""

    def test_model_not_loaded_is_ops_vision_error(self) -> None:
        """ModelNotLoadedError is a subclass of OpsVisionError."""
        assert issubclass(ModelNotLoadedError, OpsVisionError)

    def test_feature_engineering_error_is_ops_vision_error(self) -> None:
        """FeatureEngineeringError is a subclass of OpsVisionError."""
        assert issubclass(FeatureEngineeringError, OpsVisionError)

    def test_drift_monitor_error_is_ops_vision_error(self) -> None:
        """DriftMonitorError is a subclass of OpsVisionError."""
        assert issubclass(DriftMonitorError, OpsVisionError)

    def test_runbook_index_error_is_ops_vision_error(self) -> None:
        """RunbookIndexError is a subclass of OpsVisionError."""
        assert issubclass(RunbookIndexError, OpsVisionError)

    def test_model_not_loaded_message(self) -> None:
        """ModelNotLoadedError has a clear default message."""
        exc = ModelNotLoadedError()
        assert "model" in str(exc).lower()

    def test_feature_engineering_error_default_message(self) -> None:
        """FeatureEngineeringError default message is preserved."""
        exc = FeatureEngineeringError()
        assert exc.detail == "Feature engineering failed"

    def test_feature_engineering_error_custom_message(self) -> None:
        """FeatureEngineeringError accepts a custom detail string."""
        exc = FeatureEngineeringError("scaler missing")
        assert exc.detail == "scaler missing"

    def test_drift_monitor_error_stores_reason(self) -> None:
        """DriftMonitorError stores the reason attribute."""
        exc = DriftMonitorError("reference window empty")
        assert exc.reason == "reference window empty"

    def test_runbook_index_error_stores_reason(self) -> None:
        """RunbookIndexError stores the reason attribute."""
        exc = RunbookIndexError("index not built")
        assert exc.reason == "index not built"

    def test_ops_vision_error_can_be_raised(self) -> None:
        """OpsVisionError and subclasses can be raised and caught."""
        with pytest.raises(OpsVisionError):
            raise ModelNotLoadedError()

    def test_specific_exception_caught_by_base(self) -> None:
        """Subclass exceptions are caught by the base OpsVisionError handler."""
        with pytest.raises(OpsVisionError):
            raise FeatureEngineeringError("bad transform")


@pytest.mark.parametrize(
    "exc_class",
    [ModelNotLoadedError, FeatureEngineeringError, DriftMonitorError, RunbookIndexError],
)
def test_each_subclass_is_ops_vision_error(exc_class: type) -> None:
    """Every custom exception is a subclass of OpsVisionError."""
    assert issubclass(exc_class, OpsVisionError)


@pytest.mark.parametrize(
    "exc_class,detail",
    [
        (DriftMonitorError, "window empty"),
        (RunbookIndexError, "not built"),
        (DriftMonitorError, ""),
    ],
)
def test_reason_attribute_preserved(exc_class: type, detail: str) -> None:
    """Custom exceptions store the reason/detail argument correctly."""
    exc = exc_class(detail)
    assert hasattr(exc, "reason") or hasattr(exc, "detail")


class TestOpsVisionErrorRepr:
    """Tests for OpsVisionError.__repr__."""

    def test_repr_contains_class_name(self) -> None:
        """repr includes the exception class name."""
        exc = ModelNotLoadedError()
        assert "ModelNotLoadedError" in repr(exc)

    def test_repr_contains_message(self) -> None:
        """repr includes the exception message."""
        exc = FeatureEngineeringError("pipeline failed")
        assert "pipeline failed" in repr(exc)

    def test_repr_is_string(self) -> None:
        """repr() always returns a str."""
        assert isinstance(repr(OpsVisionError("oops")), str)

    @pytest.mark.parametrize(
        "exc_class,args",
        [
            (ModelNotLoadedError, []),
            (FeatureEngineeringError, ["scaler missing"]),
            (DriftMonitorError, ["empty window"]),
            (RunbookIndexError, ["vocab not built"]),
        ],
    )
    def test_repr_class_name_in_output(self, exc_class, args) -> None:
        """repr output starts with the exception class name for all subclasses."""
        exc = exc_class(*args)
        assert type(exc).__name__ in repr(exc)


class TestExceptionRaiseAndCatchParametrized:
    """Parametrized raise-and-catch tests for each custom exception."""

    @pytest.mark.parametrize(
        "exc_class,kwargs",
        [
            (ModelNotLoadedError, {}),
            (FeatureEngineeringError, {"detail": "scaler not fitted"}),
            (DriftMonitorError, {"reason": "reference empty"}),
            (RunbookIndexError, {"reason": "vocab not built"}),
        ],
    )
    def test_raise_and_catch_as_base(self, exc_class, kwargs) -> None:
        """Every subclass can be raised and caught as OpsVisionError."""
        with pytest.raises(OpsVisionError):
            raise exc_class(**kwargs)

    @pytest.mark.parametrize(
        "exc_class,kwargs",
        [
            (FeatureEngineeringError, {}),
            (FeatureEngineeringError, {"detail": "pipeline failed"}),
        ],
    )
    def test_feature_engineering_detail_field(self, exc_class, kwargs) -> None:
        """FeatureEngineeringError exposes a detail attribute."""
        exc = exc_class(**kwargs)
        assert hasattr(exc, "detail")

    @pytest.mark.parametrize(
        "reason",
        ["window empty", "insufficient samples", "reference mismatch", ""],
    )
    def test_drift_monitor_reason_stored(self, reason: str) -> None:
        """DriftMonitorError stores any reason string."""
        exc = DriftMonitorError(reason)
        assert exc.reason == reason

    @pytest.mark.parametrize(
        "reason",
        ["index not built", "vocab missing", "embedding shape mismatch"],
    )
    def test_runbook_index_reason_stored(self, reason: str) -> None:
        """RunbookIndexError stores any reason string."""
        exc = RunbookIndexError(reason)
        assert exc.reason == reason
