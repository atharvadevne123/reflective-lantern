"""Seed the Ticket-Oracle database with synthetic training data.

Run once after `alembic upgrade head` to populate the PredictionLog
table with representative entries so drift checks have a baseline.

Usage:
    python scripts/seed_db.py [--n 500]
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import PredictionLog, _get_session_factory, init_db

PRIORITIES = ["P1", "P2", "P3", "P4"]
DEPARTMENTS = ["IT", "HR", "Finance", "Operations", "Sales"]
CHANNELS = ["email", "phone", "chat", "portal"]


def seed(n: int = 500) -> None:
    init_db()
    Session = _get_session_factory()

    rng = random.Random(42)
    now = datetime.utcnow()

    with Session() as session:
        for i in range(n):
            created_at = now - timedelta(hours=rng.randint(0, 720))
            priority = rng.choices(PRIORITIES, weights=[5, 20, 50, 25])[0]
            sla_risk = rng.uniform(0.0, 1.0)
            sla_breached = priority == "P1" and sla_risk > 0.7

            log = PredictionLog(
                ticket_id=f"SEED-{i:05d}",
                predicted_priority=priority,
                priority_confidence=round(rng.uniform(0.5, 0.99), 4),
                sla_breach_risk=round(sla_risk, 4),
                sla_breached=sla_breached,
                resolution_hours_estimate=round(rng.uniform(0.5, 72.0), 2),
                department=rng.choice(DEPARTMENTS),
                channel=rng.choice(CHANNELS),
                created_at=created_at,
            )
            session.add(log)
        session.commit()
    print(f"Seeded {n} prediction log entries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Ticket-Oracle DB")
    parser.add_argument("--n", type=int, default=500, help="Number of seed rows")
    args = parser.parse_args()
    seed(args.n)


if __name__ == "__main__":
    main()
