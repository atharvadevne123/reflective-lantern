"""Seed the prediction log so drift checks have a reference window.

``run_drift_check`` compares the last 24 hours against everything older than
``REFERENCE_WINDOW_DAYS``. A freshly deployed instance has neither, so the
endpoint reports "insufficient data" until enough traffic accumulates. This
script backfills both sides so the drift path can be exercised immediately.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the project root,
# so `import app` would fail. Prepend the root to keep `python
# scripts/seed_data.py` working alongside `python -m scripts.seed_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import PredictionLog, SessionLocal, create_tables  # noqa: E402
from app.model import generate_synthetic_data  # noqa: E402

logger = logging.getLogger(__name__)


def seed(
    db: Session,
    n_reference: int = 400,
    n_recent: int = 100,
    drift: bool = False,
) -> dict[str, int]:
    """Insert reference-window and recent-window prediction rows.

    Args:
        db: Active database session.
        n_reference: Rows to write into the historical window.
        n_recent: Rows to write into the last 24 hours.
        drift: When True, the recent rows are scaled up so the KS test
            registers a shift -- useful for demonstrating the alert path.

    Returns:
        Counts of the rows written, keyed by window.
    """
    settings = get_settings()
    now = datetime.utcnow()
    ref_ts = now - timedelta(days=settings.reference_window_days + 3)
    recent_ts = now - timedelta(hours=2)

    ref_X, ref_y = generate_synthetic_data(n_reference, seed=1)
    cur_X, cur_y = generate_synthetic_data(n_recent, seed=2)

    if drift:
        # A tenfold volume shift is well beyond normal variation, so the
        # KS test should reject at any sane threshold.
        cur_X["src_bytes"] = cur_X["src_bytes"] * 10 + 5000

    for frame, labels, ts in ((ref_X, ref_y, ref_ts), (cur_X, cur_y, recent_ts)):
        for i in range(len(frame)):
            row = frame.iloc[i]
            db.add(
                PredictionLog(
                    timestamp=ts + timedelta(seconds=i),
                    src_bytes=float(row["src_bytes"]),
                    dst_bytes=float(row["dst_bytes"]),
                    duration=float(row["duration"]),
                    protocol_type=str(row["protocol_type"]),
                    service=str(row["service"]),
                    flag=str(row["flag"]),
                    prediction=str(labels.iloc[i]),
                    confidence=0.9,
                )
            )
    db.commit()

    counts = {"reference": n_reference, "recent": n_recent}
    logger.info("seeded %s (drift=%s)", counts, drift)
    return counts


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=int, default=400)
    parser.add_argument("--recent", type=int, default=100)
    parser.add_argument(
        "--drift", action="store_true", help="shift the recent window to trigger drift"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    create_tables()
    with SessionLocal() as db:
        counts = seed(db, args.reference, args.recent, args.drift)
    print(f"Seeded {counts['reference']} reference and {counts['recent']} recent rows.")
    print("Now call GET /api/v1/drift to see the KS result.")


if __name__ == "__main__":
    main()
