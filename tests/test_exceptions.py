"""Tests for app/exceptions.py."""

from __future__ import annotations

import pytest

from app.exceptions import (
    ConfigurationError,
    DatabaseError,
    DriftDetectionError,
    FeatureValidationError,
    ModelNotLoadedError,
    PredictionError,
    WattGuardError,
)


class TestWattGuardError:
    def test_is_base_exception(self) -> None:
        assert issubclass(WattGuardError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(WattGuardError):
            raise WattGuardError("test error")

    def test_message_preserved(self) -> None:
        with pytest.raises(WattGuardError, match="test message"):
            raise WattGuardError("test message")


class TestModelNotLoadedError:
    def test_is_watt_guard_error(self) -> None:
        assert issubclass(ModelNotLoadedError, WattGuardError)

    def test_can_be_caught_as_base(self) -> None:
        with pytest.raises(WattGuardError):
            raise ModelNotLoadedError("model not ready")


class TestFeatureValidationError:
    def test_stores_field_and_reason(self) -> None:
        err = FeatureValidationError("temperature_c", "must be in range [-50, 60]")
        assert err.field == "temperature_c"
        assert err.reason == "must be in range [-50, 60]"

    def test_message_contains_field(self) -> None:
        err = FeatureValidationError("occupancy", "negative value")
        assert "occupancy" in str(err)

    def test_is_watt_guard_error(self) -> None:
        assert issubclass(FeatureValidationError, WattGuardError)

    @pytest.mark.parametrize(
        "field,reason",
        [
            ("temperature_c", "out of range"),
            ("humidity_pct", "negative value"),
            ("consumption_kwh", "exceeds maximum"),
        ],
    )
    def test_parametrized_fields(self, field: str, reason: str) -> None:
        err = FeatureValidationError(field, reason)
        assert err.field == field
        assert err.reason == reason


class TestDriftDetectionError:
    def test_is_watt_guard_error(self) -> None:
        assert issubclass(DriftDetectionError, WattGuardError)


class TestDatabaseError:
    def test_is_watt_guard_error(self) -> None:
        assert issubclass(DatabaseError, WattGuardError)

    def test_can_wrap_original_exception(self) -> None:
        original = ValueError("connection refused")
        try:
            raise DatabaseError("DB failed") from original
        except DatabaseError as err:
            assert err.__cause__ is original


class TestConfigurationError:
    def test_is_watt_guard_error(self) -> None:
        assert issubclass(ConfigurationError, WattGuardError)


class TestPredictionError:
    def test_is_watt_guard_error(self) -> None:
        assert issubclass(PredictionError, WattGuardError)

    def test_message_preserved(self) -> None:
        with pytest.raises(PredictionError, match="pipeline failed"):
            raise PredictionError("pipeline failed")


def test_exception_hierarchy_catches_all_as_base() -> None:
    exceptions = [
        ModelNotLoadedError("test"),
        FeatureValidationError("f", "r"),
        DriftDetectionError("test"),
        DatabaseError("test"),
        ConfigurationError("test"),
        PredictionError("test"),
    ]
    for exc in exceptions:
        with pytest.raises(WattGuardError):
            raise exc


@pytest.mark.parametrize(
    "exc_class,args",
    [
        (ModelNotLoadedError, ("model path",)),
        (DriftDetectionError, ("feature drift detected",)),
        (DatabaseError, ("connection failed",)),
        (ConfigurationError, ("missing key",)),
        (PredictionError, ("prediction failed",)),
    ],
)
def test_exception_message_preserved(exc_class, args: tuple) -> None:
    exc = exc_class(*args)
    assert str(exc) != ""


@pytest.mark.parametrize("field,reason", [("temperature", "out of range"), ("pressure", "negative"), ("vibration", "nan")])
def test_feature_validation_error_contains_field(field: str, reason: str) -> None:
    exc = FeatureValidationError(field, reason)
    assert field in str(exc)


@pytest.mark.parametrize(
    "exc_class",
    [ModelNotLoadedError, DriftDetectionError, DatabaseError, ConfigurationError, PredictionError],
)
def test_exceptions_are_exception(exc_class) -> None:
    exc = exc_class("test")
    assert isinstance(exc, Exception)
