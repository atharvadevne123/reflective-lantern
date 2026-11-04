"""Tests for Ops-Vision Settings configuration and validators."""

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_app_name(self) -> None:
        """Default app_name is 'Ops-Vision'."""
        s = Settings()
        assert s.app_name == "Ops-Vision"

    def test_default_debug_false(self) -> None:
        """Debug mode is off by default."""
        s = Settings()
        assert s.debug is False

    def test_default_drift_threshold(self) -> None:
        """Default drift threshold is 0.05."""
        s = Settings()
        assert s.drift_threshold == 0.05

    def test_default_rate_limit_requests(self) -> None:
        """Default rate limit is 100 requests."""
        s = Settings()
        assert s.rate_limit_requests == 100

    def test_default_rate_limit_window(self) -> None:
        """Default rate limit window is 60 seconds."""
        s = Settings()
        assert s.rate_limit_window_seconds == 60

    def test_default_log_level(self) -> None:
        """Default log level is INFO."""
        s = Settings()
        assert s.log_level == "INFO"


class TestSettingsValidators:
    """Tests for Settings field validators."""

    def test_valid_drift_threshold_accepted(self) -> None:
        """drift_threshold in (0, 1) is accepted."""
        s = Settings(drift_threshold=0.01)
        assert s.drift_threshold == 0.01

    def test_drift_threshold_zero_rejected(self) -> None:
        """drift_threshold=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="drift_threshold"):
            Settings(drift_threshold=0.0)

    def test_drift_threshold_one_rejected(self) -> None:
        """drift_threshold=1 raises ValidationError."""
        with pytest.raises(ValidationError, match="drift_threshold"):
            Settings(drift_threshold=1.0)

    def test_rate_limit_requests_positive(self) -> None:
        """rate_limit_requests=10 is accepted."""
        s = Settings(rate_limit_requests=10)
        assert s.rate_limit_requests == 10

    def test_rate_limit_requests_zero_rejected(self) -> None:
        """rate_limit_requests=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_requests"):
            Settings(rate_limit_requests=0)

    def test_rate_limit_requests_negative_rejected(self) -> None:
        """Negative rate_limit_requests raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_requests"):
            Settings(rate_limit_requests=-5)

    def test_rate_limit_window_positive(self) -> None:
        """rate_limit_window_seconds=30 is accepted."""
        s = Settings(rate_limit_window_seconds=30)
        assert s.rate_limit_window_seconds == 30

    def test_rate_limit_window_zero_rejected(self) -> None:
        """rate_limit_window_seconds=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_window_seconds"):
            Settings(rate_limit_window_seconds=0)


class TestSettingsParametrized:
    """Parametrized tests for Settings field validation."""

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.10, 0.20, 0.50, 0.99])
    def test_valid_drift_thresholds_accepted(self, threshold: float) -> None:
        """Any drift_threshold strictly between 0 and 1 is valid."""
        s = Settings(drift_threshold=threshold)
        assert s.drift_threshold == threshold

    @pytest.mark.parametrize("bad_threshold", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_drift_thresholds_rejected(self, bad_threshold: float) -> None:
        """drift_threshold outside (0, 1) raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(drift_threshold=bad_threshold)

    @pytest.mark.parametrize("max_req", [1, 10, 50, 100, 1000])
    def test_valid_rate_limit_requests(self, max_req: int) -> None:
        """Positive rate_limit_requests values are accepted."""
        s = Settings(rate_limit_requests=max_req)
        assert s.rate_limit_requests == max_req

    @pytest.mark.parametrize("bad_req", [0, -1, -100])
    def test_invalid_rate_limit_requests_rejected(self, bad_req: int) -> None:
        """Non-positive rate_limit_requests raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(rate_limit_requests=bad_req)

    @pytest.mark.parametrize("window", [1, 10, 30, 60, 300])
    def test_valid_rate_limit_window_seconds(self, window: int) -> None:
        """Positive rate_limit_window_seconds values are accepted."""
        s = Settings(rate_limit_window_seconds=window)
        assert s.rate_limit_window_seconds == window

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, log_level: str) -> None:
        """All standard Python log level strings are accepted."""
        s = Settings(log_level=log_level)
        assert s.log_level == log_level
