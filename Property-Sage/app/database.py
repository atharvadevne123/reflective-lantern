"""SQLAlchemy models and session management for Property-Sage."""

import logging
import os
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./property_sage.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Prediction(Base):
    """Logged inference record for a single property valuation request."""

    __tablename__ = "predictions"

    id: int = Column(Integer, primary_key=True, index=True)
    request_id: str = Column(String(64), unique=True, index=True, nullable=False)
    bedrooms: int = Column(Integer, nullable=False)
    bathrooms: float = Column(Float, nullable=False)
    sqft: float = Column(Float, nullable=False)
    lot_size: float = Column(Float, nullable=True)
    year_built: int = Column(Integer, nullable=False)
    neighborhood: str = Column(String(128), nullable=False)
    property_type: str = Column(String(64), nullable=False)
    predicted_price: float = Column(Float, nullable=False)
    predicted_rental_yield: float = Column(Float, nullable=False)
    confidence_score: float = Column(Float, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Snapshot of model performance metrics after each training run."""

    __tablename__ = "model_metrics"

    id: int = Column(Integer, primary_key=True, index=True)
    model_version: str = Column(String(32), nullable=False)
    auc_mean: float = Column(Float, nullable=True)
    auc_std: float = Column(Float, nullable=True)
    rmse: float = Column(Float, nullable=True)
    mae: float = Column(Float, nullable=True)
    n_features: int = Column(Integer, nullable=True)
    trained_at: datetime = Column(DateTime, default=datetime.utcnow)
    notes: str = Column(Text, nullable=True)


class DriftLog(Base):
    """KS-test result for a single feature drift check."""

    __tablename__ = "drift_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    feature_name: str = Column(String(64), nullable=False)
    ks_statistic: float = Column(Float, nullable=False)
    p_value: float = Column(Float, nullable=False)
    drift_detected: int = Column(Integer, nullable=False)
    checked_at: datetime = Column(DateTime, default=datetime.utcnow)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and close it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("DB session closed")


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Initialising database schema at %s", DATABASE_URL)
    Base.metadata.create_all(bind=engine)
