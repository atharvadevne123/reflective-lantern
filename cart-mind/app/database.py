"""SQLAlchemy database models and session management."""

import logging
import os
from datetime import datetime

from sqlalchemy import (
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

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cart_mind.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String(64), index=True)
    user_id = Column(String(64), nullable=False)
    item_id = Column(String(64))
    prediction_type = Column(String(32))  # "intent" | "recommend" | "similar"
    score = Column(Float)
    model_version = Column(String(32))
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0/1
    window_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True)
    segment = Column(String(32))
    avg_order_value = Column(Float)
    purchase_count = Column(Integer)
    preferred_category = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ItemCatalog(Base):
    __tablename__ = "item_catalog"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String(64), unique=True, index=True)
    category = Column(String(64))
    subcategory = Column(String(64))
    price = Column(Float)
    avg_rating = Column(Float)
    review_count = Column(Integer)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised")


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
