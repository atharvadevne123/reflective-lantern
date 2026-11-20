"""SQLAlchemy models and session management for Forge-Guard."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./forge_guard.db",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores every inference request for drift monitoring and audit."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    temperature = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    cycle_time = Column(Float, nullable=False)
    tool_wear = Column(Float, nullable=False)
    power_consumption = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False, index=True)
    defect_probability = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False, default="1.0.0", index=True)
    correlation_id = Column(String(64), nullable=True, index=True)


class DriftReport(Base):
    """Stores KS-test drift reports per feature per monitoring window."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    feature = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    window_size = Column(Integer, nullable=False)
    model_version = Column(String(32), nullable=False)


class RetrainingRun(Base):
    """Records automated retraining pipeline executions."""

    __tablename__ = "retraining_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    trigger = Column(String(32), nullable=False, default="scheduled")
    auc_before = Column(Float, nullable=True)
    auc_after = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    status = Column(String(16), nullable=False, default="running")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not yet exist."""
    Base.metadata.create_all(bind=engine)
