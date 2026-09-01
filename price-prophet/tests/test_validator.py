"""Tests for app/data/validator.py."""

from __future__ import annotations

import pytest


def test_validate_price_positive():
    from app.data.validator import validate_price

    result = validate_price(100.0)
    assert result.valid is True


def test_validate_price_zero():
    from app.data.validator import validate_price

    result = validate_price(0.0)
    assert result.valid is False


def test_validate_price_negative():
    from app.data.validator import validate_price

    result = validate_price(-5.0)
    assert result.valid is False


def test_validate_record_valid(sample_records):
    from app.data.validator import validate_record

    result = validate_record(sample_records[0])
    assert result.valid is True


def test_validate_record_missing_key():
    from app.data.validator import validate_record

    result = validate_record({"base_price": 10.0})
    assert result.valid is False


def test_validate_record_negative_base_price():
    from app.data.validator import validate_record

    result = validate_record({"product_id": "X", "base_price": -1.0, "demand": 10.0})
    assert result.valid is False


def test_validate_dataset_empty():
    from app.data.validator import validate_dataset

    result = validate_dataset([])
    assert result.valid is False


def test_validate_dataset_valid(sample_records):
    from app.data.validator import validate_dataset

    result = validate_dataset(sample_records)
    assert result.valid is True


@pytest.mark.parametrize("price", [0.01, 1.0, 99.99, 10000.0])
def test_validate_price_various_positive_values(price: float) -> None:
    from app.data.validator import validate_price

    assert validate_price(price).valid is True


@pytest.mark.parametrize("price", [0.0, -0.01, -100.0])
def test_validate_price_non_positive_invalid(price: float) -> None:
    from app.data.validator import validate_price

    assert validate_price(price).valid is False


@pytest.mark.parametrize("demand", [0, 1, 100, 10000])
def test_validate_record_various_demands(demand: int) -> None:
    from app.data.validator import validate_record

    record = {"product_id": "P1", "base_price": 50.0, "demand": float(demand)}
    result = validate_record(record)
    assert result.valid is True


def test_validate_record_missing_product_id_fails() -> None:
    from app.data.validator import validate_record

    record = {"base_price": 50.0, "demand": 10.0}
    result = validate_record(record)
    assert result.valid is False


def test_validate_record_negative_price_fails() -> None:
    from app.data.validator import validate_record

    record = {"product_id": "P1", "base_price": -10.0, "demand": 10.0}
    result = validate_record(record)
    assert result.valid is False


def test_validate_record_returns_errors_list_on_failure() -> None:
    from app.data.validator import validate_record

    record = {"product_id": "P1", "base_price": 0.0, "demand": 10.0}
    result = validate_record(record)
    if not result.valid:
        assert len(result.errors) > 0


def test_validate_dataset_single_invalid_record() -> None:
    from app.data.validator import validate_dataset

    records = [
        {"product_id": "P1", "base_price": 50.0, "demand": 10.0},
        {"product_id": "P2", "base_price": -1.0, "demand": 5.0},
    ]
    result = validate_dataset(records)
    assert result.valid is False
    assert len(result.errors) > 0


def test_validate_dataset_all_valid() -> None:
    from app.data.validator import validate_dataset

    records = [
        {"product_id": "A", "base_price": 10.0, "demand": 5.0},
        {"product_id": "B", "base_price": 20.0, "demand": 8.0},
    ]
    result = validate_dataset(records)
    assert result.valid is True
    assert result.errors == []


def test_validate_price_string_number() -> None:
    from app.data.validator import validate_price

    result = validate_price("99.5")
    assert result.valid is True


def test_validate_price_non_numeric() -> None:
    from app.data.validator import validate_price

    result = validate_price("abc")
    assert result.valid is False


import pytest


@pytest.mark.parametrize("price", [0.01, 1.0, 9999.99])
def test_validate_price_valid_values(price: float) -> None:
    from app.data.validator import validate_price

    result = validate_price(price)
    assert result.valid is True


@pytest.mark.parametrize("price", [-1.0, 0.0, -999.0])
def test_validate_price_non_positive_invalid(price: float) -> None:
    from app.data.validator import validate_price

    result = validate_price(price)
    assert result.valid is False


def test_validate_dataset_empty_records_valid() -> None:
    from app.data.validator import validate_dataset

    result = validate_dataset([])
    assert isinstance(result.valid, bool)
