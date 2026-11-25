"""Database models and session management for Support-Sage."""
import os
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./support_sage.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TicketPrediction(Base):
    """Stores every prediction for monitoring and drift detection."""

    __tablename__ = "ticket_predictions"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(64), index=True)
    subject = Column(Text)
    description = Column(Text)
    predicted_priority = Column(String(16))
    predicted_category = Column(String(64))
    predicted_resolution_hours = Column(Float)
    priority_confidence = Column(Float)
    subject_len = Column(Integer)
    desc_len = Column(Integer)
    word_count = Column(Integer)
    has_error_keyword = Column(Integer)
    has_urgent_keyword = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class ModelMetric(Base):
    """Records model training and evaluation metrics."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(32))
    auc_mean = Column(Float)
    auc_std = Column(Float)
    n_features = Column(Integer)
    n_samples = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


def get_db():
    """Dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
