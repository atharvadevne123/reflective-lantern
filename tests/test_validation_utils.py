"""Tests for app.validation_utils module."""

from __future__ import annotations

import pytest

from app.validation_utils import (
    is_valid_email,
    is_valid_percentage,
    is_valid_temperature_c,
    is_valid_url,
    validate_positive,
    validate_range,
)


class TestValidatePositive:
    def test_positive_value_returned(self) -> None:
        assert validate_positive(5.0) == 5.0

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_positive(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_positive(-1.0)

    def test_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            validate_positive(-1.0, name="weight")

    @pytest.mark.parametrize("v", [0.001, 1.0, 100.0, 1e6])
    def test_positive_values_pass(self, v: float) -> None:
        assert validate_positive(v) == v


class TestValidateRange:
    def test_value_within_range_returned(self) -> None:
        assert validate_range(5.0, 0.0, 10.0) == 5.0

    def test_below_low_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_range(-1.0, 0.0, 10.0)

    def test_above_high_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_range(11.0, 0.0, 10.0)

    def test_at_boundaries_pass(self) -> None:
        assert validate_range(0.0, 0.0, 10.0) == 0.0
        assert validate_range(10.0, 0.0, 10.0) == 10.0

    def test_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="humidity"):
            validate_range(110.0, 0.0, 100.0, name="humidity")


class TestIsValidPercentage:
    @pytest.mark.parametrize("v", [0.0, 50.0, 100.0])
    def test_valid_percentages(self, v: float) -> None:
        assert is_valid_percentage(v) is True

    @pytest.mark.parametrize("v", [-0.1, 100.1, 200.0])
    def test_invalid_percentages(self, v: float) -> None:
        assert is_valid_percentage(v) is False


class TestIsValidTemperatureC:
    @pytest.mark.parametrize("t", [-50.0, 0.0, 20.0, 60.0])
    def test_valid_temperatures(self, t: float) -> None:
        assert is_valid_temperature_c(t) is True

    @pytest.mark.parametrize("t", [-51.0, 61.0, 200.0])
    def test_invalid_temperatures(self, t: float) -> None:
        assert is_valid_temperature_c(t) is False


class TestIsValidEmail:
    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "a.b+c@domain.org",
            "test@sub.domain.io",
        ],
    )
    def test_valid_emails(self, email: str) -> None:
        assert is_valid_email(email) is True

    @pytest.mark.parametrize(
        "email",
        [
            "notanemail",
            "@missing-local.com",
            "missing-at-sign.com",
            "double@@at.com",
            "user@",
        ],
    )
    def test_invalid_emails(self, email: str) -> None:
        assert is_valid_email(email) is False


class TestIsValidUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com",
            "https://api.example.com/v1",
            "http://localhost:8000",
        ],
    )
    def test_valid_urls(self, url: str) -> None:
        assert is_valid_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com",
            "example.com",
            "https://",
            "not-a-url",
        ],
    )
    def test_invalid_urls(self, url: str) -> None:
        assert is_valid_url(url) is False
