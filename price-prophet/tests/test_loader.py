"""Tests for app/data/loader.py."""
from __future__ import annotations


def test_generate_synthetic_data_default():
    from app.data.loader import generate_synthetic_data
    records = generate_synthetic_data()
    assert len(records) == 500


def test_generate_synthetic_data_custom_n():
    from app.data.loader import generate_synthetic_data
    records = generate_synthetic_data(n_samples=10, seed=1)
    assert len(records) == 10


def test_generate_synthetic_data_fields():
    from app.data.loader import generate_synthetic_data
    records = generate_synthetic_data(n_samples=3, seed=7)
    required = {"product_id", "base_price", "demand", "competition_price",
                "day_of_week", "is_weekend", "category", "revenue"}
    for rec in records:
        assert required.issubset(rec.keys())


def test_generate_synthetic_data_prices_positive():
    from app.data.loader import generate_synthetic_data
    records = generate_synthetic_data(n_samples=50, seed=42)
    for rec in records:
        assert rec["base_price"] > 0


def test_load_csv_missing_file():
    from app.data.loader import load_csv
    import pytest
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/data.csv")


def test_load_json_missing_file():
    from app.data.loader import load_json
    import pytest
    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/path/data.json")
