"""SQLAlchemy models and session management."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./watt_guard.db")

ADDRESS_MAX_LEN = 500
ZIPCODE_MAX_LEN = 10
CITY_MAX_LEN = 100
STATE_MAX_LEN = 50
MODEL_VERSION_MAX_LEN = 50
CORRELATION_ID_MAX_LEN = 50
FEATURE_NAME_MAX_LEN = 100

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class EnergyReading(Base):
    """Raw energy consumption reading from a building sensor."""

    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(ADDRESS_MAX_LEN))
    zipcode = Column(String(ZIPCODE_MAX_LEN), index=True)
    city = Column(String(CITY_MAX_LEN))
    state = Column(String(STATE_MAX_LEN))
    sqft = Column(Float)
    bedrooms = Column(Integer)
    bathrooms = Column(Float)
    lot_size = Column(Float)
    year_built = Column(Integer)
    renovation_year = Column(Integer, nullable=True)
    condition_score = Column(Float)
    list_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Audit log of every prediction made by the forecasting model."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, nullable=True)
    predicted_value = Column(Float)
    investment_score = Column(Float, nullable=True)
    model_version = Column(String(MODEL_VERSION_MAX_LEN))
    features_json = Column(Text)
    correlation_id = Column(String(CORRELATION_ID_MAX_LEN), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyLog(Base):
    """Record of anomaly detection results."""

    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(FEATURE_NAME_MAX_LEN))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    sample_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """KS-test drift detection result per feature."""

    __tablename__ = "drift_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    feature_name: str = Column(String(64), nullable=False)
    ks_statistic: float = Column(Float, nullable=False)
    p_value: float = Column(Float, nullable=False)
    drift_detected: int = Column(Integer, nullable=False)
    checked_at: datetime = Column(DateTime, default=datetime.utcnow)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all ORM tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")
