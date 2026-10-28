"""Pytest fixtures for News-Guard tests."""

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_news_guard.db")
os.environ.setdefault("MODEL_PATH", "/tmp/test_model.joblib")
os.environ.setdefault("METRICS_PATH", "/tmp/test_metrics.json")

from app.database import Base
from app.model import NewsGuardModel, generate_synthetic_data

TEST_DB_URL = "sqlite:///./test_news_guard.db"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestSession()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(scope="session")
def trained_model():
    df, y = generate_synthetic_data(n_samples=300)
    m = NewsGuardModel()
    m.train(df, y)
    return m


@pytest.fixture
def sample_article_df():
    return pd.DataFrame([{
        "headline": "SHOCKING: Government Hides Truth About Vaccines!!!",
        "body": "Sources say the government is hiding explosive information.",
        "source_domain": "infowars.com",
        "publication_hour": 22,
        "share_count": 50000,
    }])


@pytest.fixture
def real_article_df():
    return pd.DataFrame([{
        "headline": "Reuters: Federal Reserve holds interest rates unchanged",
        "body": "The Federal Reserve kept its benchmark interest rate steady at its June meeting.",
        "source_domain": "reuters.com",
        "publication_hour": 14,
        "share_count": 1200,
    }])


@pytest.fixture
def api_client(trained_model, db_session):
    import app.main as main_module
    from app.main import app, get_db
    main_module._model = trained_model
    main_module._reference_scores = trained_model.get_reference_features()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
