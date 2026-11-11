"""Pytest fixtures and test database setup."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.features import StockFeatureEngineer, make_sample_ohlcv
from app.main import app
from app.model import _build_pipeline, prepare_training_data

TEST_DATABASE_URL = "sqlite://"


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture(scope="function")
def db_session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def trained_pipeline():
    X, y = prepare_training_data(n_tickers=2)
    pipe = _build_pipeline()
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="session")
def sample_features():
    df = make_sample_ohlcv(100, ticker="TEST")
    eng = StockFeatureEngineer()
    eng.fit(df)
    return eng.transform(df)


@pytest.fixture(scope="session")
def client(engine):
    def override_get_db():
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    import app.main as main_module
    X, y = prepare_training_data(n_tickers=2)
    pipe = _build_pipeline()
    pipe.fit(X, y)
    main_module._model = pipe

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def settings():
    """Override settings for tests."""
    from app.config import Settings
    return Settings(DATABASE_URL="sqlite://", LOG_LEVEL="WARNING")
