"""SQLAlchemy models and session management for Ticket-Oracle."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./ticket_oracle.db")


def get_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        logger.info("database_engine_created", extra={"url": url.split("@")[-1]})
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Persists every prediction for drift analysis and auditing."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(64), index=True, nullable=False)
    priority_predicted = Column(String(4), nullable=False)
    priority_p1_prob = Column(Float, nullable=False)
    priority_p2_prob = Column(Float, nullable=False)
    priority_p3_prob = Column(Float, nullable=False)
    priority_p4_prob = Column(Float, nullable=False)
    sla_breach_risk = Column(Float, nullable=False)
    sla_breach_predicted = Column(Boolean, nullable=False)
    estimated_resolution_hours = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DriftLog(Base):
    """Stores KS-test and PSI drift check results per feature."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    psi_value = Column(Float, nullable=True)
    drift_detected = Column(Boolean, nullable=False)
    sample_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelMetrics(Base):
    """Training run history for champion/challenger tracking."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(32), nullable=False)
    auc_roc_mean = Column(Float, nullable=False)
    auc_roc_std = Column(Float, nullable=False)
    accuracy_mean = Column(Float, nullable=False)
    n_features = Column(Integer, nullable=False)
    n_samples = Column(Integer, nullable=False)
    promoted = Column(Boolean, nullable=False, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create all tables if they do not exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("database_tables_created")


def get_db():
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
