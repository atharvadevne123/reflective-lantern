"""SQLAlchemy models and session management."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./ticket_scout.db",
)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores every prediction request and result."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(64), index=True, nullable=False)
    subject = Column(Text, nullable=False)
    body_snippet = Column(Text, nullable=True)
    priority = Column(String(16), nullable=True)
    predicted_category = Column(String(64), nullable=False)
    sla_breach_prob = Column(Float, nullable=False)
    resolution_hours_pred = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(32), nullable=True)


class DriftLog(Base):
    """Stores drift check results per monitoring window."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    checked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    feature = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)  # 0/1


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all tables if they do not exist."""
    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
