"""SQLAlchemy models and session management."""
from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./alpha_scout.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class PredictionLog(Base):
    """Stores every alpha score prediction for auditing and monitoring."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    alpha_score = Column(Float, nullable=False)
    return_prediction = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """Records drift check results per feature for retrospective analysis."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)  # 0 or 1
    sample_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
