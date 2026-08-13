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
