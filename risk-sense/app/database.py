"""SQLAlchemy models and database session for Risk-Sense."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./risk_sense.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class CreditApplication(Base):
    """Stores credit risk scoring requests and results."""

    __tablename__ = "credit_applications"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(String, index=True)
    risk_score = Column(Float)
    risk_category = Column(String)
    default_probability = Column(Float)
    debt_to_income = Column(Float)
    payment_history_score = Column(Float)
    credit_utilization = Column(Float)
    account_age_months = Column(Integer)
    num_open_accounts = Column(Integer)
    num_derogatory_marks = Column(Integer)
    annual_income = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    """Tracks feature distribution drift over time."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, index=True)
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0 or 1
    logged_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
