"""SQLAlchemy models and session management."""
from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./logistics_flow.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Stores each inference call for monitoring and drift detection."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    carrier = Column(String(64))
    distance_km = Column(Float)
    weight_kg = Column(Float)
    route_type = Column(String(32))
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    predicted_minutes = Column(Float)
    confidence = Column(Float)
    model_version = Column(String(32))


class DriftLog(Base):
    """Records KS-test drift results over time."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    feature = Column(String(64))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0/1


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised")


def get_db() -> Session:
    """Yield a database session and close it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
