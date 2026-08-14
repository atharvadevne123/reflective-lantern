"""Tests for app/data/validator.py."""

from __future__ import annotations


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
