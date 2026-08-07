"""Seed the database with synthetic manufacturing sensor data for development."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def seed(n_rows: int = 500, seed_val: int = 999, batch_size: int = 100) -> int:
    """Generate synthetic predictions and insert them into the database.

    Args:
        n_rows: Number of synthetic sensor readings to insert.
        seed_val: Random seed for reproducible data generation.
        batch_size: Number of rows to commit per database transaction.

    Returns:
        Number of rows successfully inserted.
    """
    from app.database import SessionLocal, init_db
    from app.features import generate_synthetic_data
    from app.monitoring import log_prediction

    init_db()
    df = generate_synthetic_data(n_samples=n_rows, seed=seed_val)
    db = SessionLocal()
    inserted = 0

    try:
        for i, (_, row) in enumerate(df.iterrows()):
            sensor = {k: float(row[k]) for k in df.columns if k != "defect"}
            log_prediction(
                db=db,
                sensor_data=sensor,
                prediction=int(row["defect"]),
                defect_probability=float(row["defect"]) * 0.85 + 0.05,
                model_version="seed-1.0.0",
            )
            inserted += 1
            if (i + 1) % batch_size == 0:
                logger.debug("Committed batch at row %d.", i + 1)

        logger.info("Seeded %d rows into prediction_logs.", inserted)
    except Exception:
        logger.exception("Seed failed at row %d.", inserted)
        raise
    finally:
        db.close()

    return inserted


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Forge-Guard database with synthetic data")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to insert")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=100, help="Rows per transaction")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    count = seed(n_rows=args.rows, seed_val=args.seed, batch_size=args.batch_size)
    sys.exit(0 if count > 0 else 1)
