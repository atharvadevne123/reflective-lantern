"""SQLAlchemy models and database session management."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./quake_net.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SeismicEvent(Base):
    """A scored seismic event: the request features plus the model's output."""

    __tablename__ = "seismic_events"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float, nullable=False)
    station_count = Column(Integer, nullable=False)
    p_wave_amplitude = Column(Float, nullable=False)
    s_wave_amplitude = Column(Float, nullable=False)
    epicentral_distance_km = Column(Float, nullable=False)
    fault_type = Column(String(32), nullable=False)
    predicted_magnitude = Column(Float, nullable=False)
    aftershock_probability = Column(Float, nullable=True)
    model_version = Column(String(32), default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """One feature's KS-test result from a single drift check."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    sample_size = Column(Integer, nullable=False)
    checked_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Historical record of one training run's evaluation metrics."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(32), nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    r2 = Column(Float, nullable=False)
    cv_r2_mean = Column(Float, nullable=False)
    cv_r2_std = Column(Float, nullable=False)
    n_features = Column(Integer, nullable=False)
    n_samples = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a request-scoped session, closing it when the request completes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create any missing tables.

    Safe to call on every boot: existing tables are left untouched. Schema
    changes go through Alembic, not through this function.
    """
    Base.metadata.create_all(bind=engine)
