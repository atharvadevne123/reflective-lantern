"""SQLAlchemy models and session management for Cart-Mind.

Four tables back the service: ``prediction_logs`` and ``drift_logs`` capture what
the model did and how its inputs behaved, while ``user_profiles`` and
``item_catalog`` hold the entity data recommendations are drawn from. The
prediction and drift tables are what give the retraining DAG a real feedback
loop instead of a synthetic one.
"""

import logging
import os
from collections.abc import Generator
from datetime import UTC, datetime

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


def _utcnow() -> datetime:
    """Return an aware UTC timestamp (datetime.utcnow is deprecated in 3.12)."""
    return datetime.now(UTC)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cart_mind.db")

_is_sqlite = "sqlite" in DATABASE_URL
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    # Connection pool tuning for PostgreSQL; ignored by SQLite's StaticPool.
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """One row per scored request, for monitoring and retraining feedback."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String(64), index=True)
    user_id = Column(String(64), nullable=False)
    item_id = Column(String(64))
    prediction_type = Column(String(32))  # "intent" | "recommend" | "similar"
    score = Column(Float)
    model_version = Column(String(32))
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=_utcnow)


class DriftLog(Base):
    """One row per feature per drift check, with the KS statistic and verdict."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0/1
    window_size = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)


class UserProfile(Base):
    """Aggregated per-user shopping behaviour used to build request features."""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True)
    segment = Column(String(32))
    avg_order_value = Column(Float)
    purchase_count = Column(Integer)
    preferred_category = Column(String(64))
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class ItemCatalog(Base):
    """Product catalogue backing candidate generation and similarity search."""

    __tablename__ = "item_catalog"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String(64), unique=True, index=True)
    category = Column(String(64))
    subcategory = Column(String(64))
    price = Column(Float)
    avg_rating = Column(Float)
    review_count = Column(Integer)
    description = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


def init_db() -> None:
    """Create every table that does not yet exist.

    Safe to call repeatedly — ``create_all`` skips existing tables. Alembic owns
    schema changes in production; this is the convenience path for local runs.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised")


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped database session, closing it on completion.

    Yields:
        An open SQLAlchemy session, closed when the request finishes.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
