"""SQLAlchemy models and session management for Volt-Cast."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./volt_cast.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    load_mw = Column(Float, nullable=False)
    temperature_c = Column(Float, nullable=True)
    humidity_pct = Column(Float, nullable=True)
    hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, default=False)
    region = Column(String(64), default="default")
    source = Column(String(64), default="api")
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    predicted_load_mw = Column(Float, nullable=False)
    actual_load_mw = Column(Float, nullable=True)
    model_version = Column(String(32), default="1.0.0")
    region = Column(String(64), default="default")
    features_json = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    request_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    feature_name = Column(String(128), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, default=False)
    reference_window = Column(Integer, default=1000)
    current_window = Column(Integer, default=200)
    model_version = Column(String(32), default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class RetrainingLog(Base):
    __tablename__ = "retraining_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    trigger = Column(String(64), default="manual")
    samples_used = Column(Integer, nullable=False)
    auc_before = Column(Float, nullable=True)
    auc_after = Column(Float, nullable=True)
    rmse_after = Column(Float, nullable=True)
    r2_after = Column(Float, nullable=True)
    status = Column(String(32), default="success")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("database tables created")


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_recent_predictions(db: Session, limit: int = 200) -> list[PredictionLog]:
    """Return the most recent prediction log entries."""
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )


def get_drift_summary(db: Session, model_version: str = "1.0.0") -> dict:
    """Aggregate drift statistics for the given model version."""
    reports = (
        db.query(DriftReport)
        .filter(DriftReport.model_version == model_version)
        .all()
    )
    if not reports:
        return {"total": 0, "drifted": 0, "drift_pct": 0.0}
    drifted = sum(1 for r in reports if r.drift_detected)
    return {
        "total": len(reports),
        "drifted": drifted,
        "drift_pct": round(100 * drifted / len(reports), 1),
    }
