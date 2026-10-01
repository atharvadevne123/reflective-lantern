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


@pytest.mark.parametrize(
    "field,reason",
    [("temperature", "out of range"), ("pressure", "negative"), ("vibration", "nan")],
)
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


def test_database_error_repr_contains_message() -> None:
    err = DatabaseError("timeout after 30s")
    assert "timeout" in str(err)


def test_configuration_error_can_be_chained() -> None:
    cause = KeyError("missing_key")
    try:
        raise ConfigurationError("bad config") from cause
    except ConfigurationError as err:
        assert err.__cause__ is cause


@pytest.mark.parametrize("msg", ["short", "a" * 500])
def test_prediction_error_various_lengths(msg: str) -> None:
    err = PredictionError(msg)
    assert msg in str(err)


@pytest.mark.parametrize(
    "exc_class",
    [
        WattGuardError,
        ModelNotLoadedError,
        FeatureValidationError,
        DriftDetectionError,
        DatabaseError,
        ConfigurationError,
        PredictionError,
    ],
)
def test_all_exceptions_inherit_watt_guard_error(exc_class: type) -> None:
    """Every domain exception is a WattGuardError."""
    assert issubclass(exc_class, WattGuardError)


@pytest.mark.parametrize("msg", ["msg1", "hello world", "x" * 100])
def test_watt_guard_error_str_matches_message(msg: str) -> None:
    err = WattGuardError(msg)
    assert msg in str(err)


def test_feature_validation_error_has_field_and_reason_attrs() -> None:
    err = FeatureValidationError(field="temperature", reason="out of range")
    assert err.field == "temperature"
    assert err.reason == "out of range"


class TestRateLimitExceededError:
    def test_is_watt_guard_error(self) -> None:
        from app.exceptions import RateLimitExceededError

        assert issubclass(RateLimitExceededError, WattGuardError)

    def test_stores_limit_and_retry_after(self) -> None:
        from app.exceptions import RateLimitExceededError

        err = RateLimitExceededError(limit=100, retry_after_seconds=30)
        assert err.limit == 100
        assert err.retry_after_seconds == 30

    def test_message_contains_limit(self) -> None:
        from app.exceptions import RateLimitExceededError

        err = RateLimitExceededError(limit=50, retry_after_seconds=10)
        assert "50" in str(err)

    @pytest.mark.parametrize("limit,retry", [(10, 5), (100, 60), (1000, 0)])
    def test_various_limits(self, limit: int, retry: int) -> None:
        from app.exceptions import RateLimitExceededError

        err = RateLimitExceededError(limit=limit, retry_after_seconds=retry)
        assert err.limit == limit


class TestExternalServiceError:
    def test_is_watt_guard_error(self) -> None:
        from app.exceptions import ExternalServiceError

        assert issubclass(ExternalServiceError, WattGuardError)

    def test_stores_service_and_reason(self) -> None:
        from app.exceptions import ExternalServiceError

        err = ExternalServiceError("payment-api", "503 Service Unavailable")
        assert err.service == "payment-api"
        assert err.reason == "503 Service Unavailable"

    def test_message_contains_service(self) -> None:
        from app.exceptions import ExternalServiceError

        err = ExternalServiceError("weather-api", "timeout")
        assert "weather-api" in str(err)

    @pytest.mark.parametrize(
        "service,reason",
        [
            ("db", "connection refused"),
            ("cache", "eviction error"),
            ("ml-server", "model not found"),
        ],
    )
    def test_various_services(self, service: str, reason: str) -> None:
        from app.exceptions import ExternalServiceError

        err = ExternalServiceError(service, reason)
        assert err.service == service
        assert err.reason == reason


def test_model_not_loaded_error_str_is_non_empty() -> None:
    """ModelNotLoadedError str representation is non-empty."""
    err = ModelNotLoadedError("no model found")
    assert len(str(err)) > 0


def test_drift_detection_error_message() -> None:
    """DriftDetectionError message is preserved in str()."""
    msg = "feature drift detected in temperature_c"
    err = DriftDetectionError(msg)
    assert msg in str(err)


def test_prediction_error_is_not_model_not_loaded() -> None:
    """PredictionError and ModelNotLoadedError are distinct exception types."""
    assert PredictionError is not ModelNotLoadedError
    assert not issubclass(PredictionError, ModelNotLoadedError)


@pytest.mark.parametrize("exc_class", [DatabaseError, ConfigurationError, PredictionError])
def test_exception_can_be_raised_and_caught_by_type(exc_class) -> None:
    """Each exception subclass can be raised and caught by its own type."""
    with pytest.raises(exc_class):
        raise exc_class("test message")


def test_feature_validation_error_both_attrs_accessible() -> None:
    """FeatureValidationError exposes both field and reason without error."""
    err = FeatureValidationError("humidity_pct", "value out of [0, 100]")
    assert isinstance(err.field, str)
    assert isinstance(err.reason, str)
    assert err.field == "humidity_pct"
    assert err.reason == "value out of [0, 100]"
