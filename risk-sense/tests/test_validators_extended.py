"""Extended validator tests for risk-sense with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.validators import (
    validate_application,
    validate_dti,
    validate_income,
    validate_positive_integer,
    validate_probability_field,
)


@pytest.mark.parametrize("valid_dti", [0.0, 0.5, 1.0, 5.0, 10.0])
def test_validate_dti_valid_values(valid_dti: float) -> None:
    assert validate_dti(valid_dti) == valid_dti


@pytest.mark.parametrize("bad_dti", [-0.001, -1.0, 10.001, 100.0])
def test_validate_dti_invalid_raises(bad_dti: float) -> None:
    with pytest.raises(ValueError, match="debt_to_income"):
        validate_dti(bad_dti)


@pytest.mark.parametrize(
    "valid_prob,field",
    [
        (0.0, "payment_history_score"),
        (0.5, "credit_utilization"),
        (1.0, "payment_history_score"),
        (0.9999, "credit_utilization"),
    ],
)
def test_validate_probability_field_valid(valid_prob: float, field: str) -> None:
    result = validate_probability_field(valid_prob, field)
    assert result == valid_prob


@pytest.mark.parametrize(
    "bad_prob,field",
    [
        (-0.001, "payment_history_score"),
        (1.001, "credit_utilization"),
        (-1.0, "payment_history_score"),
        (2.0, "credit_utilization"),
    ],
)
def test_validate_probability_field_invalid_raises(bad_prob: float, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        validate_probability_field(bad_prob, field)


@pytest.mark.parametrize(
    "val,field",
    [
        (0, "account_age_months"),
        (1, "num_open_accounts"),
        (100, "num_derogatory_marks"),
        (999, "account_age_months"),
    ],
)
def test_validate_positive_integer_valid(val: int, field: str) -> None:
    assert validate_positive_integer(val, field) == val


@pytest.mark.parametrize(
    "bad_val,field",
    [
        (-1, "account_age_months"),
        (-100, "num_open_accounts"),
    ],
)
def test_validate_positive_integer_negative_raises(bad_val: int, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        validate_positive_integer(bad_val, field)


@pytest.mark.parametrize("valid_income", [0.0, 1.0, 50_000.0, 1_000_000.0])
def test_validate_income_valid(valid_income: float) -> None:
    assert validate_income(valid_income) == valid_income


@pytest.mark.parametrize("bad_income", [-0.01, -1000.0])
def test_validate_income_negative_raises(bad_income: float) -> None:
    with pytest.raises(ValueError, match="annual_income"):
        validate_income(bad_income)


def _valid_application(**overrides) -> dict:
    base = {
        "debt_to_income": 0.3,
        "payment_history_score": 0.85,
        "credit_utilization": 0.4,
        "account_age_months": 60,
        "num_open_accounts": 3,
        "num_derogatory_marks": 0,
        "annual_income": 75_000.0,
    }
    base.update(overrides)
    return base


def test_validate_application_valid() -> None:
    result = validate_application(_valid_application())
    assert result["debt_to_income"] == pytest.approx(0.3)


def test_validate_application_bad_dti_raises() -> None:
    with pytest.raises(ValueError):
        validate_application(_valid_application(debt_to_income=15.0))


def test_validate_application_bad_payment_score_raises() -> None:
    with pytest.raises(ValueError):
        validate_application(_valid_application(payment_history_score=1.5))


def test_validate_application_bad_credit_utilization_raises() -> None:
    with pytest.raises(ValueError):
        validate_application(_valid_application(credit_utilization=-0.1))


def test_validate_application_bad_income_raises() -> None:
    with pytest.raises(ValueError):
        validate_application(_valid_application(annual_income=-100.0))


def test_validate_application_bad_account_age_raises() -> None:
    with pytest.raises(ValueError):
        validate_application(_valid_application(account_age_months=-1))


def test_validate_application_returns_dict() -> None:
    result = validate_application(_valid_application())
    assert isinstance(result, dict)


def test_validate_application_zero_income_valid() -> None:
    result = validate_application(_valid_application(annual_income=0.0))
    assert result["annual_income"] == 0.0


def test_validate_application_zero_dti_valid() -> None:
    result = validate_application(_valid_application(debt_to_income=0.0))
    assert result["debt_to_income"] == 0.0
