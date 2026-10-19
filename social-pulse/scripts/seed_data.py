"""Seed the database with synthetic predictions for local development."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from app.database import SessionLocal, init_db
from app.model import generate_synthetic_data
from app.monitoring import log_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed(n: int = 200) -> None:
    """Insert *n* synthetic predictions into the database.

    Args:
        n: Number of synthetic samples to insert.
    """
    init_db()
    db = SessionLocal()
    samples, labels = generate_synthetic_data(n=n)
    logger.info("Seeding %d predictions...", n)
    try:
        for sample, label in zip(samples, labels):
            log_prediction(
                db=db,
                request_id=str(uuid.uuid4()),
                sample=sample,
                viral_probability=float(label) * 0.8 + 0.1,
                viral_label=bool(label),
                ensemble_confidence=0.6,
                model_version="seed-1.0",
                latency_ms=5.0,
            )
        logger.info("Seed complete: %d predictions inserted", n)
    finally:
        db.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the seed script."""
    parser = argparse.ArgumentParser(description="Seed the Social-Pulse database")
    parser.add_argument("-n", "--count", type=int, default=200, help="Number of records to insert")
    args = parser.parse_args(argv)
    seed(n=args.count)


if __name__ == "__main__":
    main(sys.argv[1:])
