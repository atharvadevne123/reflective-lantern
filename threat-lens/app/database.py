"""SQLAlchemy models and session management for Threat-Lens."""

import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./threat_lens.db",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores every prediction request for drift monitoring and auditing."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    correlation_id = Column(String(64), index=True)
    src_bytes = Column(Float, nullable=True)
    dst_bytes = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    protocol_type = Column(String(16), nullable=True)
    service = Column(String(32), nullable=True)
    flag = Column(String(16), nullable=True)
    predicted_class = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    is_attack = Column(Integer, nullable=False)  # 0 = normal, 1 = attack
    raw_features = Column(Text, nullable=True)


class DriftReport(Base):
    """KS-test drift reports computed during monitoring runs."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    feature_name = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)
    reference_n = Column(Integer, nullable=True)
    current_n = Column(Integer, nullable=True)


class RetrainingEvent(Base):
    """Records every automated retraining trigger."""

    __tablename__ = "retraining_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    trigger_reason = Column(String(128), nullable=False)
    auc_before = Column(Float, nullable=True)
    auc_after = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    success = Column(Integer, nullable=False, default=1)


def get_db() -> Session:
    """Yield a database session and close it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not yet exist."""
    logger.info("Initialising database schema")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready")
