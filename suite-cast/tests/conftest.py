"""Shared pytest fixtures for Suite-Cast test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure suite-cast package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force SQLite test database before any app module imports settings
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_suite_cast.db")
os.environ.setdefault("MODEL_VERSION", "test-1.0.0")
os.environ.setdefault("BASE_ROOM_RATE", "150.0")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from app.database import Base  # noqa: E402


@pytest.fixture(scope="session")
def db_engine():
    """In-memory SQLite engine for the test session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Transactional database session — rolls back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_booking_df() -> pd.DataFrame:
    """Single-row booking DataFrame using representative mid-season values."""
    return pd.DataFrame(
        [
            {
                "lead_time": 14,
                "length_of_stay": 3,
                "guests_count": 2,
                "checkin_month": 7,
                "checkin_dayofweek": 4,
                "is_weekend": 1,
                "current_occ_rate": 0.75,
                "prev_year_occ_rate": 0.70,
                "room_rate": 189.0,
                "competitor_avg_rate": 175.0,
                "special_event": 0,
                "room_type": "deluxe",
                "booking_channel": "online",
            }
        ]
    )


@pytest.fixture
def sample_booking_dict() -> dict:
    """Dict version of sample_booking_df for API payload tests."""
    return {
        "lead_time": 14,
        "length_of_stay": 3,
        "guests_count": 2,
        "checkin_month": 7,
        "checkin_dayofweek": 4,
        "is_weekend": 1,
        "current_occ_rate": 0.75,
        "prev_year_occ_rate": 0.70,
        "room_rate": 189.0,
        "competitor_avg_rate": 175.0,
        "special_event": 0,
        "room_type": "deluxe",
        "booking_channel": "online",
    }


@pytest.fixture(scope="module")
def trained_models():
    """Train a small XGBoost + LightGBM ensemble once per module."""
    from app.model import generate_sample_data, train_model

    X, y = generate_sample_data(n=300)
    xgb_pipe, lgbm_pipe, metrics = train_model(X, y)
    return xgb_pipe, lgbm_pipe, metrics


@pytest.fixture
def reference_scores(trained_models) -> list[float]:
    """Generate reference demand scores from the trained model."""
    from app.model import generate_sample_data

    xgb_pipe, lgbm_pipe, _ = trained_models
    X_ref, _ = generate_sample_data(100)
    probs = (xgb_pipe.predict_proba(X_ref)[:, 1] + lgbm_pipe.predict_proba(X_ref)[:, 1]) / 2.0
    return probs.tolist()
