"""Pytest fixtures for Sports-Oracle tests."""

from __future__ import annotations

import os
from collections.abc import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_sports_oracle.db")

from app.database import Base
from app.features import make_synthetic_dataset

TEST_DB_URL = "sqlite:///./test_sports_oracle.db"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    import os as _os
    try:
        _os.remove("test_sports_oracle.db")
    except FileNotFoundError:
        pass


@pytest.fixture()
def db_session(test_engine) -> Generator[Session, None, None]:
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def sample_features() -> pd.DataFrame:
    X, _ = make_synthetic_dataset(n=100)
    return X


@pytest.fixture(scope="session")
def trained_model(sample_features, tmp_path_factory):
    from app.features import make_synthetic_dataset
    from app.model import train_model

    tmp = tmp_path_factory.mktemp("model")
    X, y = make_synthetic_dataset(n=500)
    pipe, metrics = train_model(X, y, model_path=tmp / "model.joblib", metrics_path=tmp / "metrics.json")
    return pipe, metrics, tmp


@pytest.fixture(scope="session")
def api_client(trained_model):
    pipe, _, model_dir = trained_model
    import shutil
    shutil.copy(model_dir / "model.joblib", "model.joblib")
    shutil.copy(model_dir / "metrics.json", "metrics.json")

    import app.main as main_module
    from app.main import app
    main_module._model = pipe

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    import os
    for f in ("model.joblib", "metrics.json", "test_sports_oracle.db"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


@pytest.fixture()
def sample_match_payload() -> dict:
    return {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "competition": "Premier League",
        "home_form": 0.7,
        "away_form": 0.5,
        "home_attack": 1.4,
        "away_attack": 1.2,
        "home_defense": 1.3,
        "away_defense": 1.1,
        "h2h_home_wins": 5,
        "h2h_draws": 3,
        "h2h_away_wins": 2,
        "home_rest_days": 7,
        "away_rest_days": 5,
    }
