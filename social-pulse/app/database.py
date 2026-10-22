"""SQLAlchemy models and database session management."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    JSON,
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

DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./social_pulse.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Log of every inference request."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), unique=True, index=True, nullable=False)
    text = Column(Text, nullable=False)
    author_followers = Column(Integer, nullable=False)
    has_media = Column(Boolean, default=False)
    has_link = Column(Boolean, default=False)
    hashtag_count = Column(Integer, default=0)
    mention_count = Column(Integer, default=0)
    hour_of_day = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    viral_probability = Column(Float, nullable=False)
    viral_label = Column(Boolean, nullable=False)
    ensemble_confidence = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    feature_vector = Column(JSON, nullable=True)


class DriftRecord(Base):
    """KS-test drift detection results."""

    __tablename__ = "drift_records"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    sample_size = Column(Integer, nullable=False)
    checked_at = Column(DateTime, default=datetime.utcnow, index=True)


class TrainingRun(Base):
    """Metadata for each model training run."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, nullable=False)
    model_version = Column(String(32), nullable=False)
    auc_roc = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    train_samples = Column(Integer, nullable=False)
    feature_names = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(32), default="running")


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and close it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
