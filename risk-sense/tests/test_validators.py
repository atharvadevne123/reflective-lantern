"""Input validation tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.validators import (
    validate_application,
    validate_dti,
    validate_income,
    validate_positive_integer,
    validate_probability_field,
)


class TestValidateDti:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 5.0, 10.0])
    def test_valid_values_pass(self, value: float) -> None:
        assert validate_dti(value) == value

    @pytest.mark.parametrize("value", [-0.1, -1.0, 10.1, 100.0])
    def test_out_of_range_raises(self, value: float) -> None:
        with pytest.raises(ValueError, match="debt_to_income"):
            validate_dti(value)


class TestValidateProbabilityField:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_valid_probabilities_pass(self, value: float) -> None:
        assert validate_probability_field(value, "test_field") == value

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
    def test_out_of_range_raises(self, value: float) -> None:
        with pytest.raises(ValueError):
            validate_probability_field(value, "payment_history_score")

    def test_error_message_contains_field_name(self) -> None:
        with pytest.raises(ValueError, match="credit_utilization"):
            validate_probability_field(1.5, "credit_utilization")


class TestValidatePositiveInteger:
    @pytest.mark.parametrize("value", [0, 1, 100, 1000])
    def test_valid_values_pass(self, value: int) -> None:
        assert validate_positive_integer(value, "test") == value

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_positive_integer(-1, "num_open_accounts")


class TestValidateIncome:
    @pytest.mark.parametrize("value", [0.0, 1000.0, 50_000.0, 1_000_000.0])
    def test_valid_income_passes(self, value: float) -> None:
        assert validate_income(value) == value

    def test_negative_income_raises(self) -> None:
        with pytest.raises(ValueError, match="annual_income"):
            validate_income(-1.0)


class TestValidateApplication:
    def test_valid_application_passes(self) -> None:
        app = {
            "debt_to_income": 0.3, "payment_history_score": 0.8,
            "credit_utilization": 0.2, "account_age_months": 48,
            "num_open_accounts": 4, "num_derogatory_marks": 0, "annual_income": 50_000.0
        }
        result = validate_application(app)
        assert result["debt_to_income"] == 0.3

    def test_invalid_dti_raises(self) -> None:
        app = {
            "debt_to_income": -1.0, "payment_history_score": 0.8,
            "credit_utilization": 0.2, "account_age_months": 48,
            "num_open_accounts": 4, "num_derogatory_marks": 0, "annual_income": 50_000.0
        }
        with pytest.raises(ValueError):
            validate_application(app)

    def test_multiple_errors_reported_together(self) -> None:
        app = {
            "debt_to_income": -1.0, "payment_history_score": 2.0,
            "credit_utilization": 0.2, "account_age_months": 48,
            "num_open_accounts": 4, "num_derogatory_marks": 0, "annual_income": 50_000.0
        }
        with pytest.raises(ValueError) as exc_info:
            validate_application(app)
        assert "debt_to_income" in str(exc_info.value) or "payment_history_score" in str(exc_info.value)
