"""Pytest fixtures and test database configuration."""
from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_sport_cast.db")

from app.database import Base, get_db
from app.features import make_synthetic_dataset
from app.main import app

TEST_DB_URL = "sqlite:///./test_sport_cast.db"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(test_engine):
    TestingSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client(db_session):
    def override_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_match_payload() -> dict:
    return {
        "match_id": "test-match-001",
        "home_team": "Team Alpha",
        "away_team": "Team Beta",
        "home_wins_last5": 3,
        "home_draws_last5": 1,
        "home_losses_last5": 1,
        "away_wins_last5": 2,
        "away_draws_last5": 1,
        "away_losses_last5": 2,
        "home_goals_avg": 1.8,
        "home_goals_conceded_avg": 1.1,
        "away_goals_avg": 1.4,
        "away_goals_conceded_avg": 1.5,
        "h2h_home_wins": 3,
        "h2h_away_wins": 2,
        "h2h_draws": 1,
        "home_ranking": 5.0,
        "away_ranking": 12.0,
        "home_elo": 1650.0,
        "away_elo": 1520.0,
        "home_days_rest": 7,
        "away_days_rest": 4,
        "home_is_home_ground": 1.0,
    }


@pytest.fixture()
def synthetic_dataset():
    return make_synthetic_dataset(n=500, seed=0)


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame([{
        "home_wins_last5": 3, "home_draws_last5": 1, "home_losses_last5": 1,
        "away_wins_last5": 2, "away_draws_last5": 1, "away_losses_last5": 2,
        "home_goals_avg": 1.8, "home_goals_conceded_avg": 1.1,
        "away_goals_avg": 1.4, "away_goals_conceded_avg": 1.5,
        "h2h_home_wins": 3, "h2h_away_wins": 2, "h2h_draws": 1,
        "home_ranking": 5.0, "away_ranking": 12.0,
        "home_elo": 1650.0, "away_elo": 1520.0,
        "home_days_rest": 7, "away_days_rest": 4,
        "home_is_home_ground": 1.0,
    }])
