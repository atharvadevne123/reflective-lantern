"""Seed the database with synthetic neighbourhood stats and sample properties.

Run once after `init_db()` to populate reference data used by the
/neighborhood-stats endpoint and by the FAISS index warm-up.

Usage
-----
    python scripts/seed_data.py
"""

import logging
import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base, NeighborhoodStat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ZIPCODES = [
    ("94102", 1_200_000, 800.0, 8.5, 9.0, 8.5, 0.28, 0.052),
    ("94110", 980_000, 680.0, 7.5, 8.0, 8.0, 0.32, 0.055),
    ("94117", 1_450_000, 950.0, 9.0, 9.5, 9.0, 0.18, 0.048),
    ("90210", 4_500_000, 1200.0, 9.5, 5.0, 7.5, 0.10, 0.030),
    ("10001", 2_200_000, 1400.0, 8.0, 9.5, 9.5, 0.25, 0.035),
    ("10022", 3_800_000, 1800.0, 8.5, 9.5, 9.5, 0.15, 0.028),
    ("60601", 450_000, 280.0, 7.5, 8.5, 7.5, 0.38, 0.068),
    ("77001", 280_000, 180.0, 6.5, 6.0, 5.5, 0.45, 0.078),
    ("98101", 820_000, 550.0, 8.0, 8.5, 8.5, 0.22, 0.058),
    ("30301", 420_000, 260.0, 7.0, 7.5, 6.5, 0.40, 0.072),
]


def seed(db_url: str | None = None) -> None:
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///./realty_edge.db")
    engine = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        inserted = 0
        for (zipcode, med_price, med_ppsf, school, transit, walk, crime, rental) in ZIPCODES:
            existing = db.query(NeighborhoodStat).filter(NeighborhoodStat.zipcode == zipcode).first()
            if existing:
                continue
            stat = NeighborhoodStat(
                zipcode=zipcode,
                median_price=med_price,
                median_price_per_sqft=med_ppsf,
                school_score=school,
                transit_score=transit,
                walkability_score=walk,
                crime_rate=crime,
                avg_rental_yield=rental,
            )
            db.add(stat)
            inserted += 1
        db.commit()
        logger.info("Seeded %d neighbourhood records.", inserted)
    finally:
        db.close()


if __name__ == "__main__":
    seed()
