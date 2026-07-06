"""Pytest fixtures and test database setup."""

import numpy as np
import pandas as pd
import pytest
from app.database import Base, get_db
from app.main import app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DB_URL = "sqlite:///./test_realty_edge.db"

_test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def db_session():
    db = TestSessionLocal()
    yield db
    db.close()


@pytest.fixture
def client(setup_test_db):
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_property():
    return {
        "sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "lot_size": 5000.0,
        "year_built": 1990,
        "renovation_year": 2015,
        "condition_score": 7.5,
        "zipcode": "94102",
        "city": "San Francisco",
        "state": "CA",
        "school_score": 8.0,
        "transit_score": 9.0,
        "walkability_score": 8.5,
        "crime_rate": 0.3,
        "median_neighborhood_price": 1_200_000.0,
        "median_price_per_sqft": 800.0,
        "avg_rental_yield": 0.05,
        "listing_days": 14,
        "list_price": 1_100_000.0,
    }


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "sqft": rng.uniform(800, 4000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
        "lot_size": rng.uniform(2000, 15000, n),
        "year_built": rng.integers(1950, 2020, n),
        "renovation_year": rng.integers(2000, 2025, n),
        "condition_score": rng.uniform(4, 10, n),
        "zipcode": ["94102"] * n,
        "city": ["San Francisco"] * n,
        "state": ["CA"] * n,
        "school_score": rng.uniform(3, 10, n),
        "transit_score": rng.uniform(3, 10, n),
        "walkability_score": rng.uniform(3, 10, n),
        "crime_rate": rng.uniform(0.1, 0.8, n),
        "median_neighborhood_price": rng.uniform(300_000, 2_000_000, n),
        "median_price_per_sqft": rng.uniform(150, 1000, n),
        "avg_rental_yield": rng.uniform(0.03, 0.10, n),
        "listing_days": rng.integers(1, 365, n),
        "list_price": rng.uniform(300_000, 2_000_000, n),
    })


@pytest.fixture
def sample_target(sample_df):
    rng = np.random.default_rng(0)
    base = sample_df["sqft"] * sample_df["median_price_per_sqft"]
    return (base + rng.normal(0, 50_000, len(sample_df))).clip(100_000).values
