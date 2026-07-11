"""SQLAlchemy models and session management."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./watt_guard.db")

_is_sqlite = "sqlite" in DATABASE_URL
_engine_kwargs: dict[str, object] = {"echo": False}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
    _engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    _engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class EnergyReading(Base):
    """Raw energy consumption reading from a building sensor."""

    __tablename__ = "energy_readings"

    id: int = Column(Integer, primary_key=True, index=True)
    building_id: str = Column(String(64), index=True, nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False, index=True)
    consumption_kwh: float = Column(Float, nullable=False)
    temperature_c: float = Column(Float)
    humidity_pct: float = Column(Float)
    occupancy: int = Column(Integer)
    hvac_state: int = Column(Integer, default=0)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Audit log of every prediction made by the forecasting model."""

    __tablename__ = "prediction_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    building_id: str = Column(String(64), index=True)
    timestamp: datetime = Column(DateTime, nullable=False, index=True)
    predicted_kwh: float = Column(Float, nullable=False)
    actual_kwh: float = Column(Float)
    model_version: str = Column(String(32), default="1.0.0")
    latency_ms: float = Column(Float)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, index=True)


class AnomalyLog(Base):
    """Record of anomaly detection results."""

    __tablename__ = "anomaly_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    building_id: str = Column(String(64), index=True)
    timestamp: datetime = Column(DateTime, nullable=False)
    consumption_kwh: float = Column(Float, nullable=False)
    anomaly_score: float = Column(Float, nullable=False)
    is_anomaly: int = Column(Integer, nullable=False)
    severity: str = Column(String(16))
    created_at: datetime = Column(DateTime, default=datetime.utcnow)


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
