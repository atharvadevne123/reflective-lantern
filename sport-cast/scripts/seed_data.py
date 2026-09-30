"""Seed Sport-Cast database with synthetic sample data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import MatchRecord, init_db, _get_session_factory
from datetime import datetime, timedelta
import random

def seed(n: int = 20) -> None:
    init_db()
    Session = _get_session_factory()
    db = Session()
    teams = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "ManUtd", "Spurs", "Brighton", "Newcastle"]
    try:
        for i in range(n):
            home, away = random.sample(teams, 2)
            db.add(MatchRecord(
                match_id=f"seed-{i:04d}",
                home_team=home,
                away_team=away,
                sport="football",
                match_date=datetime.utcnow() - timedelta(days=i * 7),
                result=random.choice(["home_win", "draw", "away_win"]),
            ))
        db.commit()
        print(f"Seeded {n} match records.")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
