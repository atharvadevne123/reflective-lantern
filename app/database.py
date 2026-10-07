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
    checked_at = Column(DateTime, nullable=True)
    feature = Column(String(64))
    feature_name = Column(String(64))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0/1


class EnergyReading(Base):
    """Raw energy consumption reading from a building sensor."""

    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(128), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    consumption_kwh = Column(Float)
    temperature_c = Column(Float, nullable=True)
    humidity_pct = Column(Float, nullable=True)
    occupancy = Column(Integer, nullable=True)
    hvac_state = Column(Integer, nullable=True)


class PredictionLog(Base):
    """Stores per-building energy consumption predictions."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(128), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    predicted_kwh = Column(Float)
    actual_kwh = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    model_version = Column(String(32), default="1.0.0")


class AnomalyLog(Base):
    """Records detected anomalies in energy consumption."""

    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(128), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    consumption_kwh = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Integer)  # 0/1
    severity = Column(String(32), nullable=True)


class ModelMetrics(Base):
    """Stores model evaluation metrics after each training run."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(32))
    r2_mean = Column(Float, nullable=True)
    mae_kwh = Column(Float, nullable=True)
    rmse_mean = Column(Float, nullable=True)


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


def get_predictions_by_building(
    db: Session, building_id: str, limit: int = 100
) -> list[PredictionLog]:
    """Return prediction log entries for a given building, newest first."""
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.building_id == building_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_recent_anomalies(
    db: Session,
    building_id: str,
    limit: int = 50,
    severity: str | None = None,
) -> list[AnomalyLog]:
    """Return recent anomaly log entries, optionally filtered by severity."""
    q = db.query(AnomalyLog).filter(AnomalyLog.building_id == building_id)
    if severity is not None:
        q = q.filter(AnomalyLog.severity == severity)
    return q.order_by(AnomalyLog.timestamp.desc()).limit(limit).all()


def count_anomalies_by_building(db: Session, building_id: str) -> int:
    """Return the total number of anomaly records for a building."""
    return db.query(AnomalyLog).filter(AnomalyLog.building_id == building_id).count()


__all__ = [
    "AnomalyLog",
    "Base",
    "DATABASE_URL",
    "DriftLog",
    "EnergyReading",
    "ModelMetrics",
    "Prediction",
    "PredictionLog",
    "SessionLocal",
    "count_anomalies_by_building",
    "engine",
    "get_db",
    "get_predictions_by_building",
    "get_recent_anomalies",
    "init_db",
]
