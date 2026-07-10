"""SQLAlchemy models and session management."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./watt_guard.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class EnergyReading(Base):
    """Raw energy consumption readings."""

    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    consumption_kwh = Column(Float, nullable=False)
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    occupancy = Column(Integer)
    hvac_state = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Log of every prediction made by the API."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(64), index=True)
    timestamp = Column(DateTime, nullable=False)
    predicted_kwh = Column(Float, nullable=False)
    actual_kwh = Column(Float)
    model_version = Column(String(32), default="1.0.0")
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyLog(Base):
    """Log of detected anomalies."""

    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(String(64), index=True)
    timestamp = Column(DateTime, nullable=False)
    consumption_kwh = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Integer, nullable=False)
    severity = Column(String(16))
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """KS-test drift detection results."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)
    checked_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
