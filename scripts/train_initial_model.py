"""Train and save an initial model from synthetic data.

Generates a realistic synthetic real estate dataset and trains the ensemble.
Run this before starting the API if no `model.joblib` exists.

Usage
-----
    python scripts/train_initial_model.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.model import MODEL_VERSION, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N = 2000
RNG = np.random.default_rng(42)

ZIPCODES = ["94102", "90210", "10001", "60601", "77001", "98101"]


def generate_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    sqft = RNG.uniform(600, 5000, N)
    beds = RNG.integers(1, 6, N)
    baths = RNG.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], N)
    lot = RNG.uniform(1000, 20000, N)
    built = RNG.integers(1940, 2023, N)
    reno = np.where(RNG.random(N) > 0.6, RNG.integers(2000, 2025, N), built)
    condition = RNG.uniform(3, 10, N)
    school = RNG.uniform(3, 10, N)
    transit = RNG.uniform(2, 10, N)
    walk = RNG.uniform(2, 10, N)
    crime = RNG.uniform(0.05, 0.70, N)
    med_price = RNG.uniform(200_000, 2_500_000, N)
    med_ppsf = med_price / RNG.uniform(1200, 3500, N)
    rental_yield = RNG.uniform(0.03, 0.11, N)
    listing_days = RNG.integers(1, 400, N)
    zipcode = RNG.choice(ZIPCODES, N)

    base_val = (sqft * med_ppsf * 0.7 +
                condition * 15_000 +
                school * 20_000 -
                crime * 80_000 +
                RNG.normal(0, 30_000, N))
    base_val = base_val.clip(50_000, 10_000_000)

    df = pd.DataFrame({
        "sqft": sqft, "bedrooms": beds, "bathrooms": baths, "lot_size": lot,
        "year_built": built, "renovation_year": reno, "condition_score": condition,
        "zipcode": zipcode, "city": "City", "state": "CA",
        "school_score": school, "transit_score": transit, "walkability_score": walk,
        "crime_rate": crime, "median_neighborhood_price": med_price,
        "median_price_per_sqft": med_ppsf, "avg_rental_yield": rental_yield,
        "listing_days": listing_days, "list_price": base_val * 1.05,
    })
    return df, base_val


if __name__ == "__main__":
    logger.info("Generating %d synthetic property records...", N)
    df, y = generate_dataset()
    logger.info("Training ensemble model version=%s...", MODEL_VERSION)
    bundle, metrics = train_model(df, y)
    logger.info("Training complete: R2=%.4f RMSE=%.0f", metrics["r2_mean"], metrics["rmse_mean"])
    logger.info("Model saved to model.joblib")
