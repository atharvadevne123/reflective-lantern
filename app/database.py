"""SQLAlchemy models and session management for prediction logging."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./traffic_pulse.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores each prediction request and its result for monitoring."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    route_id = Column(String(64), nullable=False, index=True)
    road_type = Column(String(32), nullable=False)
    congestion_level = Column(Integer, nullable=False)
    congestion_prob = Column(Float, nullable=False)
    incident_score = Column(Float, nullable=False)
    model_version = Column(String(32), default="1.0.0", nullable=False)
    features_json = Column(Text, nullable=True)


class DriftLog(Base):
    """Stores KS-test drift detection results per feature."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    feature_name = Column(String(64), nullable=False, index=True)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, default=0, nullable=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and guarantee it is closed afterwards.

    Yields:
        An active SQLAlchemy session bound to the configured engine.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not already exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
