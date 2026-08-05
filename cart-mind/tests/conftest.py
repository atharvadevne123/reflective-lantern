"""Pytest configuration, fixtures, and test database setup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.features import INTERACTION_COLS, ITEM_COLS, USER_COLS, make_sample_dataframe
from app.main import app

TEST_DB_URL = "sqlite:///./test_cart_mind.db"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
_FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db_session():
    db = TestSessionLocal()
    yield db
    db.close()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return make_sample_dataframe(n=100, seed=0)


@pytest.fixture()
def feature_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return sample_df[_FEATURE_COLS]


@pytest.fixture()
def binary_labels(sample_df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.integers(0, 2, len(sample_df)), name="purchased")


@pytest.fixture()
def trained_pipeline(feature_df, binary_labels):
    from app.model import train_model

    pipe, _ = train_model(feature_df, binary_labels)
    return pipe


@pytest.fixture()
def intent_payload() -> dict:
    return {
        "user_id": "u_001",
        "item_id": "i_abc",
        "user_age": 35,
        "purchase_count": 12,
        "avg_order_value": 120.0,
        "days_since_last_purchase": 14,
        "days_since_registration": 365,
        "session_count_7d": 8,
        "cart_abandon_rate": 0.2,
        "item_price": 49.99,
        "item_avg_rating": 4.2,
        "item_review_count": 320,
        "item_inventory_level": 85,
        "item_discount_pct": 10.0,
        "view_count": 5,
        "click_count": 2,
        "wishlist_flag": 1,
        "same_category_purchases": 4,
    }


@pytest.fixture()
def recommend_payload() -> dict:
    return {
        "user_id": "u_001",
        "user_age": 35,
        "purchase_count": 12,
        "avg_order_value": 120.0,
        "days_since_last_purchase": 14,
        "days_since_registration": 365,
        "session_count_7d": 8,
        "cart_abandon_rate": 0.2,
        "top_k": 5,
    }


@pytest.fixture()
def similar_payload() -> dict:
    return {
        "item_id": "i_abc",
        "top_k": 5,
        "item_price": 49.99,
        "item_avg_rating": 4.2,
        "item_review_count": 320,
        "item_inventory_level": 85,
        "item_discount_pct": 10.0,
    }
