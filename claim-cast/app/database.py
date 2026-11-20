"""SQLAlchemy models and session management."""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./claim_cast.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ClaimPrediction(Base):
    """Persisted claim predictions for drift monitoring."""

    __tablename__ = "claim_predictions"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String, index=True)
    severity_score = Column(Float, nullable=False)
    severity_label = Column(String, nullable=False)
    triage_priority = Column(Integer, nullable=False)
    model_version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """KS-test drift detection records."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, default=0)
    sample_size = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Create and return a new database session."""
    return SessionLocal()


def count_predictions(db: Session) -> int:
    """Return total prediction count."""
    return db.query(ClaimPrediction).count()
