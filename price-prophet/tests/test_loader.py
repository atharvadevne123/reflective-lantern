"""Tests for app/data/loader.py."""

from __future__ import annotations

import pytest


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
    required = {
        "product_id",
        "base_price",
        "demand",
        "competition_price",
        "day_of_week",
        "is_weekend",
        "category",
        "revenue",
    }
    for rec in records:
        assert required.issubset(rec.keys())


def test_generate_synthetic_data_prices_positive():
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=50, seed=42)
    for rec in records:
        assert rec["base_price"] > 0


def test_load_csv_missing_file():
    import pytest

    from app.data.loader import load_csv

    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/data.csv")


def test_load_json_missing_file():
    import pytest

    from app.data.loader import load_json

    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/path/data.json")


@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_generate_synthetic_data_count(n_samples: int) -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=n_samples)
    assert len(records) == n_samples


@pytest.mark.parametrize("n_samples", [5, 20])
def test_generate_synthetic_data_has_required_keys(n_samples: int) -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=n_samples)
    required_keys = {"product_id", "base_price", "demand"}
    for record in records:
        assert required_keys.issubset(set(record.keys()))


def test_generate_synthetic_data_reproducible() -> None:
    from app.data.loader import generate_synthetic_data

    records_a = generate_synthetic_data(n_samples=10, seed=99)
    records_b = generate_synthetic_data(n_samples=10, seed=99)
    assert records_a[0]["base_price"] == records_b[0]["base_price"]


def test_generate_synthetic_data_base_price_positive() -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=50)
    assert all(r["base_price"] > 0.0 for r in records)


def test_generate_synthetic_data_demand_non_negative() -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=50)
    assert all(r["demand"] >= 0.0 for r in records)


def test_load_csv_file_not_found_raises() -> None:
    from app.data.loader import load_csv

    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/data.csv")


def test_load_json_file_not_found_raises() -> None:
    from app.data.loader import load_json

    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/path/data.json")


@pytest.mark.parametrize("n", [1, 10, 100])
def test_generate_synthetic_data_count(n: int) -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=n)
    assert len(records) == n


def test_generate_synthetic_data_has_product_id() -> None:
    from app.data.loader import generate_synthetic_data

    records = generate_synthetic_data(n_samples=5)
    assert all("product_id" in r for r in records)
