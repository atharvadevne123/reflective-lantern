"""Tests for app/data/augmentor.py."""

from __future__ import annotations


def test_add_noise_length():
    from app.data.augmentor import add_noise

    result = add_noise([10.0, 20.0, 30.0])
    assert len(result) == 3


def test_add_noise_zero_seed_deterministic():
    from app.data.augmentor import add_noise

    a = add_noise([5.0, 10.0], seed=0)
    b = add_noise([5.0, 10.0], seed=0)
    assert a == b


def test_bootstrap_sample_length(sample_records):
    from app.data.augmentor import bootstrap_sample

    result = bootstrap_sample(sample_records, n=10, seed=1)
    assert len(result) == 10


def test_bootstrap_sample_empty():
    from app.data.augmentor import bootstrap_sample

    assert bootstrap_sample([], n=5) == []


def test_augment_dataset_size(sample_records):
    from app.data.augmentor import augment_dataset

    result = augment_dataset(sample_records, multiplier=3)
    assert len(result) == len(sample_records) * 3


def test_augment_dataset_multiplier_one(sample_records):
    from app.data.augmentor import augment_dataset

    result = augment_dataset(sample_records, multiplier=1)
    assert len(result) == len(sample_records)


def test_add_noise_preserves_length() -> None:
    from app.data.augmentor import add_noise

    values = [10.0, 20.0, 30.0]
    result = add_noise(values, noise_fraction=0.05)
    assert len(result) == len(values)


def test_add_noise_empty_returns_empty() -> None:
    from app.data.augmentor import add_noise

    assert add_noise([]) == []


def test_bootstrap_sample_fixed_size() -> None:
    from app.data.augmentor import bootstrap_sample

    records = [{"base_price": float(i)} for i in range(10)]
    result = bootstrap_sample(records, n=5)
    assert len(result) == 5


def test_augment_dataset_multiplier_zero_returns_originals() -> None:
    from app.data.augmentor import augment_dataset

    records = [{"base_price": 100.0, "demand": 10.0}]
    result = augment_dataset(records, multiplier=0)
    assert len(result) == len(records)


def test_augment_dataset_preserves_non_numeric_fields() -> None:
    from app.data.augmentor import augment_dataset

    records = [{"base_price": 50.0, "product_id": "P1", "category": "electronics"}]
    result = augment_dataset(records, multiplier=2)
    for rec in result:
        assert rec.get("product_id") == "P1"
        assert rec.get("category") == "electronics"


def test_add_noise_different_seeds_produce_different_results() -> None:
    from app.data.augmentor import add_noise

    a = add_noise([10.0, 20.0, 30.0], seed=0)
    b = add_noise([10.0, 20.0, 30.0], seed=99)
    assert a != b


def test_bootstrap_sample_all_from_source() -> None:
    from app.data.augmentor import bootstrap_sample

    source = [{"id": i} for i in range(5)]
    result = bootstrap_sample(source, n=20)
    source_ids = {r["id"] for r in source}
    for rec in result:
        assert rec["id"] in source_ids
