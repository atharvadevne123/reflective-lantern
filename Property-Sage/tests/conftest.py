"""pytest fixtures for Property-Sage test suite."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_property_sage.db"

test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_property():
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft": 1500.0,
        "lot_size": 6000.0,
        "year_built": 2005,
        "neighborhood": "suburb",
        "property_type": "house",
    }


@pytest.fixture
def sample_price_series():
    import numpy as np
    rng = np.random.default_rng(42)
    return list(rng.normal(loc=450_000, scale=50_000, size=100))


@pytest.fixture
def sample_sqft_series():
    import numpy as np
    rng = np.random.default_rng(42)
    return list(rng.normal(loc=1500, scale=300, size=100))
