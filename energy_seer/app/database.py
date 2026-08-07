"""SQLAlchemy database models and session management."""

import os
from datetime import datetime

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
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./energy_seer.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    meter_id = Column(String(64), index=True)
    consumption_kwh = Column(Float, nullable=False)
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    is_holiday = Column(Boolean, default=False)
    building_type = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    meter_id = Column(String(64))
    predicted_kwh = Column(Float)
    actual_kwh = Column(Float, nullable=True)
    model_version = Column(String(32))
    features_used = Column(JSON)
    confidence_score = Column(Float)
    prediction_horizon_h = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyLog(Base):
    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    meter_id = Column(String(64))
    consumption_kwh = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    anomaly_type = Column(String(32))
    severity = Column(String(16))
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    feature_name = Column(String(64))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    reference_mean = Column(Float)
    current_mean = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> None:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
