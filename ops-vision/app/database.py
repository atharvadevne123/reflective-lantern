"""SQLAlchemy models and database session management for Ops-Vision."""

import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://ops:ops@localhost:5432/opsvision"
)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _engine_kwargs(url: str) -> dict:
    """Return dialect-appropriate create_engine keyword arguments.

    SQLite (used in tests) does not support the QueuePool sizing options that
    PostgreSQL uses, so they are only applied for non-SQLite URLs.

    Args:
        url: The SQLAlchemy database URL.

    Returns:
        Dict of keyword arguments suitable for create_engine.
    """
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "echo": False,
    }


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, creating it on first use.

    The engine is created lazily so that importing this module does not
    require a database driver to be installed or a server to be reachable.

    Returns:
        The shared Engine instance.
    """
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL", DATABASE_URL)
        _engine = create_engine(url, **_engine_kwargs(url))
        logger.info("Database engine created for dialect %s", _engine.dialect.name)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the process-wide session factory, creating it on first use.

    Returns:
        A configured sessionmaker bound to the lazy engine.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Incident(Base):
    """Represents an SRE incident record."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String(128), nullable=False, index=True)
    cpu_usage_pct = Column(Float, nullable=False)
    memory_usage_pct = Column(Float, nullable=False)
    error_rate_per_min = Column(Float, nullable=False)
    latency_p99_ms = Column(Float, nullable=False)
    request_rate_per_sec = Column(Float, nullable=False)
    disk_io_util_pct = Column(Float, nullable=False)
    is_incident = Column(Boolean, nullable=False)
    severity = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # list_incidents() filters on service_name and orders by created_at, so the
    # composite index lets a single scan satisfy both halves of that query.
    __table_args__ = (
        Index("ix_incidents_service_created", "service_name", "created_at"),
    )


class Prediction(Base):
    """Stores model predictions for monitoring and drift analysis."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, nullable=True, index=True)
    service_name = Column(String(128), nullable=False)
    features = Column(JSON, nullable=False)
    predicted_incident = Column(Boolean, nullable=False)
    predicted_severity = Column(String(16), nullable=True)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # count_incidents_predicted() aggregates on the incident flag;
    # delete_old_predictions() and per-service queries use service_name + created_at.
    __table_args__ = (
        Index("ix_predictions_incident_flag", "predicted_incident"),
        Index("ix_predictions_service_created", "service_name", "created_at"),
    )


class DriftAlert(Base):
    """Records KS-test drift detection results."""

    __tablename__ = "drift_alerts"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drifted = Column(Boolean, nullable=False)
    reference_window = Column(String(32), nullable=False)
    current_window = Column(String(32), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # count_drift_alerts_last_24h() filters on drifted AND a created_at range.
    __table_args__ = (
        Index("ix_drift_alerts_drifted_created", "drifted", "created_at"),
    )


def get_db() -> Session:
    """Yield a database session and ensure it is closed after use.

    Intended for use as a FastAPI dependency.

    Yields:
        An active SQLAlchemy Session.
    """
    db = get_session_factory()()
    try:
        yield db
    except Exception:
        logger.exception("Database session error — rolling back")
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables defined by the ORM models."""
    logger.info("Creating database tables")
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables created successfully")
