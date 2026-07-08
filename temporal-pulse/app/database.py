"""SQLAlchemy models and session management for Temporal-Pulse."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./temporal_pulse.db")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SensorReading(Base):
    """Stores raw multivariate sensor readings."""

    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    values = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_sensor_timestamp", "sensor_id", "timestamp"),
    )


class AnomalyEvent(Base):
    """Stores detected anomaly events."""

    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    root_cause = Column(Text)
    features_snapshot = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_anomaly_sensor_ts", "sensor_id", "timestamp"),
    )


class Prediction(Base):
    """Stores forecast predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(64), nullable=False, index=True)
    horizon = Column(Integer, nullable=False)
    predicted_values = Column(JSON, nullable=False)
    confidence_interval = Column(JSON)
    model_version = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DriftLog(Base):
    """Stores KS-test drift detection results."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(64), nullable=False, index=True)
    feature_name = Column(String(128), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_db() -> Session:
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables: %s", e)
        raise
