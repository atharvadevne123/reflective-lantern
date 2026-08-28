"""Shared fixtures for Price-Prophet tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_records():
    """Return a small list of product record dicts for use in tests."""
    return [
        {
            "product_id": "A1",
            "base_price": 100.0,
            "demand": 80.0,
            "competition_price": 95.0,
            "day_of_week": 1,
            "is_weekend": False,
            "category": "electronics",
        },
        {
            "product_id": "A2",
            "base_price": 50.0,
            "demand": 200.0,
            "competition_price": 52.0,
            "day_of_week": 6,
            "is_weekend": True,
            "category": "clothing",
        },
        {
            "product_id": "A3",
            "base_price": 200.0,
            "demand": 30.0,
            "competition_price": 180.0,
            "day_of_week": 3,
            "is_weekend": False,
            "category": "electronics",
        },
        {
            "product_id": "A4",
            "base_price": 25.0,
            "demand": 500.0,
            "competition_price": 24.0,
            "day_of_week": 0,
            "is_weekend": False,
            "category": "food",
        },
        {
            "product_id": "A5",
            "base_price": 75.0,
            "demand": 120.0,
            "competition_price": 80.0,
            "day_of_week": 5,
            "is_weekend": True,
            "category": "clothing",
        },
    ]


@pytest.fixture
def feature_matrix():
    """Return a small feature matrix (list of lists) for model tests."""
    return [
        [100.0, 80.0, 95.0, 1.0, 0.0],
        [50.0, 200.0, 52.0, 6.0, 1.0],
        [200.0, 30.0, 180.0, 3.0, 0.0],
    ]


@pytest.fixture
def price_series():
    """Return a short sequence of sample prices for backtesting tests."""
    return [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0]


@pytest.fixture
def demand_series():
    """Return a short sequence of sample demand values for backtesting tests."""
    return [200.0, 180.0, 190.0, 170.0, 150.0, 155.0, 140.0, 120.0]
