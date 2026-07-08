"""SQLAlchemy models and session management for Cyber-Sentinel."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cyber:sentinel@localhost:5432/cybersentinel"
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class NetworkEvent(Base):
    """Stores raw network events submitted for analysis."""

    __tablename__ = "network_events"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    src_ip = Column(String(45), nullable=True)
    dst_ip = Column(String(45), nullable=True)
    protocol = Column(String(16), nullable=True)
    payload_bytes = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Float, nullable=False, default=0.0)
    flags = Column(String(64), nullable=True)
    raw_features = Column(Text, nullable=True)


class Prediction(Base):
    """Stores model predictions with confidence scores."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    label = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    attack_type = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=True)
    drift_score = Column(Float, nullable=True)


class DriftLog(Base):
    """Records KS-test drift detection results over time."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False, default=0)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    connect_args={"connect_timeout": 10} if "postgresql" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised")
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        raise


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
