"""SQLAlchemy models and session management for Suite-Cast."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON

from .config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Persisted record of each demand-score prediction."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    input_features = Column(JSON, nullable=False)
    demand_score = Column(Float, nullable=False)
    suggested_rate = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False)


class ModelMetrics(Base):
    """Training metrics snapshot stored after each retraining run."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(32), nullable=False, index=True)
    auc_mean = Column(Float, nullable=False)
    auc_std = Column(Float, nullable=False)
    n_features = Column(Integer, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    """Yield a DB session; close on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)
