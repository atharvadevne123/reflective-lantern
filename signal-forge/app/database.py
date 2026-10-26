"""SQLAlchemy models and session management for Signal-Forge."""

import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./signal_forge.db")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class MarketRegimePrediction(Base):
    """Stores market regime detection predictions."""

    __tablename__ = "market_regime_predictions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    regime = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    volatility = Column(Float)
    momentum = Column(Float)
    correlation = Column(Float)
    volume_ratio = Column(Float)
    beta = Column(Float)
    features_json = Column(Text)
    model_version = Column(String(20), default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftEvent(Base):
    """Stores data drift detection events."""

    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(100), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, default=0)
    reference_mean = Column(Float)
    current_mean = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables: %s", e)
        raise
