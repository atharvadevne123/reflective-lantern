"""SQLAlchemy models, session management, and database initialisation."""

from __future__ import annotations

import logging
from collections.abc import Generator
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
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = (
            {"check_same_thread": False}
            if settings.database_url.startswith("sqlite")
            else {}
        )
        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class VehiclePrediction(Base):
    """Stores each telemetry prediction result for audit and drift analysis."""

    __tablename__ = "vehicle_predictions"

    id: int = Column(Integer, primary_key=True, index=True)
    vehicle_id: str = Column(String(64), nullable=False, index=True)
    created_at: datetime = Column(DateTime, server_default=func.now(), nullable=False)

    # Input telemetry
    speed_kmh: float = Column(Float, nullable=False)
    engine_temp_c: float = Column(Float, nullable=False)
    oil_pressure_bar: float = Column(Float, nullable=False)
    fuel_consumption_lph: float = Column(Float, nullable=False)
    battery_voltage_v: float = Column(Float, nullable=False)
    brake_pad_mm: float = Column(Float, nullable=False)
    tire_pressure_bar: float = Column(Float, nullable=False)
    rpm: float = Column(Float, nullable=False)
    coolant_temp_c: float = Column(Float, nullable=False)
    transmission_temp_c: float = Column(Float, nullable=False)

    # Prediction outputs
    maintenance_needed: bool = Column(Boolean, nullable=False)
    anomaly_score: float = Column(Float, nullable=False)
    anomaly_type: str = Column(String(64), nullable=True)
    remaining_useful_life_km: float = Column(Float, nullable=True)
    confidence: float = Column(Float, nullable=False)
    model_version: str = Column(String(32), nullable=False)


class DriftReport(Base):
    """Records each KS-test drift check result."""

    __tablename__ = "drift_reports"

    id: int = Column(Integer, primary_key=True, index=True)
    created_at: datetime = Column(DateTime, server_default=func.now(), nullable=False)
    feature_name: str = Column(String(64), nullable=False)
    ks_statistic: float = Column(Float, nullable=False)
    p_value: float = Column(Float, nullable=False)
    drift_detected: bool = Column(Boolean, nullable=False)
    sample_size: int = Column(Integer, nullable=False)
    details: str = Column(Text, nullable=True)


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=_get_engine())
    logger.info("Database tables initialised")


def get_db() -> Generator[Session, None, None]:
    """Yield a database session."""
    factory = _get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
