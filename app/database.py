"""SQLAlchemy models and session management for Realty-Edge.

Provides ORM models for properties, prediction logs, drift reports, and
neighbourhood stats. Exports a session factory and a FastAPI dependency.
"""

import logging
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

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./realty_edge.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all Realty-Edge ORM models."""


class Property(Base):
    """ORM model for a stored property listing."""

    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(500))
    zipcode = Column(String(10), index=True)
    city = Column(String(100))
    state = Column(String(50))
    sqft = Column(Float)
    bedrooms = Column(Integer)
    bathrooms = Column(Float)
    lot_size = Column(Float)
    year_built = Column(Integer)
    renovation_year = Column(Integer, nullable=True)
    condition_score = Column(Float)
    list_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """ORM model for a single model prediction record."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, nullable=True)
    predicted_value = Column(Float)
    investment_score = Column(Float, nullable=True)
    model_version = Column(String(50))
    features_json = Column(Text)
    correlation_id = Column(String(50), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    """ORM model for a KS-test drift detection report."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(100))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    sample_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class NeighborhoodStat(Base):
    """ORM model for aggregated neighbourhood statistics by ZIP code."""

    __tablename__ = "neighborhood_stats"

    id = Column(Integer, primary_key=True, index=True)
    zipcode = Column(String(10), unique=True, index=True)
    median_price = Column(Float)
    median_price_per_sqft = Column(Float)
    school_score = Column(Float)
    transit_score = Column(Float)
    walkability_score = Column(Float)
    crime_rate = Column(Float)
    avg_rental_yield = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a database session and close it when the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that do not yet exist.

    Safe to call on startup; existing tables are left untouched.
    """
    logger.info("Initialising database schema at %s", DATABASE_URL.split("@")[-1])
    Base.metadata.create_all(bind=engine)
    logger.info("Database ready.")
