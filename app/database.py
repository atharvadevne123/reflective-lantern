"""SQLAlchemy models and session management."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from datetime import datetime

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


class ModelMetrics(Base):
    """Snapshot of model training metrics after each retrain."""

    __tablename__ = "model_metrics"

    id: int = Column(Integer, primary_key=True, index=True)
    model_version: str = Column(String(32), nullable=False, index=True)
    r2_mean: float = Column(Float)
    r2_std: float = Column(Float)
    mae_kwh: float = Column(Float)
    rmse_mean: float = Column(Float)
    n_samples: int = Column(Integer)
    n_features: int = Column(Integer)
    trained_at: datetime = Column(DateTime, default=datetime.utcnow, index=True)


def get_predictions_by_building(
    db: Session,
    building_id: str,
    limit: int = 100,
) -> list[PredictionLog]:
    """Return the most recent prediction log entries for a specific building.

    Args:
        db: Active SQLAlchemy session.
        building_id: Building identifier to filter on.
        limit: Maximum number of rows to return.

    Returns:
        List of PredictionLog ORM objects, newest first.
    """
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.building_id == building_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )


def get_recent_anomalies(
    db: "Session",
    building_id: str,
    limit: int = 10,
    severity: str | None = None,
) -> list["AnomalyLog"]:
    """Return the most recent anomaly log entries for a building.

    Args:
        db: Active SQLAlchemy session.
        building_id: Building identifier to filter on.
        limit: Maximum number of rows to return (default 10).
        severity: Optional severity filter ('low', 'medium', 'critical').

    Returns:
        List of AnomalyLog ORM objects, newest first.
    """
    query = (
        db.query(AnomalyLog)
        .filter(AnomalyLog.building_id == building_id)
        .filter(AnomalyLog.is_anomaly == 1)
    )
    if severity is not None:
        query = query.filter(AnomalyLog.severity == severity)
    return query.order_by(AnomalyLog.timestamp.desc()).limit(limit).all()


def count_anomalies_by_building(db: "Session", building_id: str) -> int:
    """Return the total number of anomaly records for *building_id*.

    Args:
        db: Active SQLAlchemy session.
        building_id: Building identifier to count anomalies for.

    Returns:
        Integer count of rows where is_anomaly == 1.
    """
    return (
        db.query(AnomalyLog)
        .filter(AnomalyLog.building_id == building_id, AnomalyLog.is_anomaly == 1)
        .count()
    )
