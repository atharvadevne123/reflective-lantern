"""
Data loading utilities for Price-Prophet.

Provides helpers to read CSV and JSON files from disk, and a synthetic
data generator that requires only the Python standard library so that
the system can be exercised without any external dataset.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from typing import Any

logger = logging.getLogger(__name__)


def load_csv(path: str) -> list[dict[str, Any]]:
    """Read a CSV file and return its rows as a list of dicts.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    list[dict]
        One dict per data row; keys are the header column names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    logger.debug("load_csv loaded %d rows from %s", len(rows), path)
    return rows


def load_json(path: str) -> list[dict[str, Any]]:
    """Read a JSON file and return its contents as a list of dicts.

    The file must contain either a JSON array of objects, or a single
    object which is wrapped in a list before returning.

    Parameters
    ----------
    path:
        Absolute or relative path to the JSON file.

    Returns
    -------
    list[dict]

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    ValueError
        If the file content cannot be parsed as JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    result = [data] if isinstance(data, dict) else list(data)
    logger.debug("load_json loaded %d records from %s", len(result), path)
    return result


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_CATEGORIES = ["electronics", "clothing", "grocery", "home", "sports", "toys"]


def generate_synthetic_data(
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a synthetic e-commerce pricing dataset using only stdlib.

    Each record represents a single product observation and contains the
    fields expected by the downstream preprocessor and feature modules.

    Parameters
    ----------
    n_samples:
        Number of records to generate (default 500).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Records with keys: product_id, base_price, demand,
        competition_price, day_of_week, is_weekend, category, revenue.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for i in range(n_samples):
        category = rng.choice(_CATEGORIES)

        # Price ranges differ by category to add realistic variation
        if category == "electronics":
            base_price = rng.uniform(50.0, 1500.0)
        elif category == "clothing":
            base_price = rng.uniform(10.0, 300.0)
        elif category == "grocery":
            base_price = rng.uniform(1.0, 50.0)
        elif category == "home":
            base_price = rng.uniform(20.0, 800.0)
        elif category == "sports":
            base_price = rng.uniform(15.0, 600.0)
        else:  # toys
            base_price = rng.uniform(5.0, 200.0)

        base_price = round(base_price, 2)

        # Competition price is correlated with base price
        competition_price = round(base_price * rng.uniform(0.8, 1.2), 2)

        day_of_week = rng.randint(0, 6)
        is_weekend = day_of_week >= 5

        # Demand is inversely related to relative price (elasticity model)
        price_ratio = base_price / max(competition_price, 0.01)
        base_demand = rng.uniform(20.0, 500.0)
        # Elastic response: higher relative price → lower demand
        elasticity_factor = math.exp(-1.5 * math.log(max(price_ratio, 0.01)))
        demand = max(0.0, round(base_demand * elasticity_factor * rng.uniform(0.85, 1.15), 2))

        # Weekend uplift
        if is_weekend:
            demand = round(demand * rng.uniform(1.05, 1.25), 2)

        revenue = round(base_price * demand, 2)

        records.append(
            {
                "product_id": f"PROD-{i + 1:05d}",
                "base_price": base_price,
                "demand": demand,
                "competition_price": competition_price,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "category": category,
                "revenue": revenue,
            }
        )

    return records
