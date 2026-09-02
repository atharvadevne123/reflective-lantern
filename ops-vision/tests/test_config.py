"""Tests for Ops-Vision Settings configuration and validators."""

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_app_name(self):
        """Default app_name is 'Ops-Vision'."""
        s = Settings()
        assert s.app_name == "Ops-Vision"

    def test_default_debug_false(self):
        """Debug mode is off by default."""
        s = Settings()
        assert s.debug is False

    def test_default_drift_threshold(self):
        """Default drift threshold is 0.05."""
        s = Settings()
        assert s.drift_threshold == 0.05

    def test_default_rate_limit_requests(self):
        """Default rate limit is 100 requests."""
        s = Settings()
        assert s.rate_limit_requests == 100

    def test_default_rate_limit_window(self):
        """Default rate limit window is 60 seconds."""
        s = Settings()
        assert s.rate_limit_window_seconds == 60

    def test_default_log_level(self):
        """Default log level is INFO."""
        s = Settings()
        assert s.log_level == "INFO"


class TestSettingsValidators:
    """Tests for Settings field validators."""

    def test_valid_drift_threshold_accepted(self):
        """drift_threshold in (0, 1) is accepted."""
        s = Settings(drift_threshold=0.01)
        assert s.drift_threshold == 0.01

    def test_drift_threshold_zero_rejected(self):
        """drift_threshold=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="drift_threshold"):
            Settings(drift_threshold=0.0)

    def test_drift_threshold_one_rejected(self):
        """drift_threshold=1 raises ValidationError."""
        with pytest.raises(ValidationError, match="drift_threshold"):
            Settings(drift_threshold=1.0)

    def test_rate_limit_requests_positive(self):
        """rate_limit_requests=10 is accepted."""
        s = Settings(rate_limit_requests=10)
        assert s.rate_limit_requests == 10

    def test_rate_limit_requests_zero_rejected(self):
        """rate_limit_requests=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_requests"):
            Settings(rate_limit_requests=0)

    def test_rate_limit_requests_negative_rejected(self):
        """Negative rate_limit_requests raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_requests"):
            Settings(rate_limit_requests=-5)

    def test_rate_limit_window_positive(self):
        """rate_limit_window_seconds=30 is accepted."""
        s = Settings(rate_limit_window_seconds=30)
        assert s.rate_limit_window_seconds == 30

    def test_rate_limit_window_zero_rejected(self):
        """rate_limit_window_seconds=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_limit_window_seconds"):
            Settings(rate_limit_window_seconds=0)
