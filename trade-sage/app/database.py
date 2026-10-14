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

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trade_sage.db")

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(DATABASE_URL, connect_args=connect_args)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(16), index=True, nullable=False)
    prediction = Column(Integer, nullable=False)  # 1=bullish, 0=bearish
    confidence = Column(Float, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    features_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    checked_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    auc_mean = Column(Float, nullable=False)
    auc_std = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    n_features = Column(Integer, nullable=False)
    n_samples = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    db = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=_get_engine())
