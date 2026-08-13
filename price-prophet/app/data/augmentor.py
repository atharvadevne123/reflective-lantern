"""
Data augmentation utilities for Price-Prophet.

Augmentation improves model robustness by expanding small datasets with
perturbed copies.  All functions use only the Python standard library.
"""

from __future__ import annotations

import random
from typing import Any


def add_noise(
    values: list[float],
    noise_fraction: float = 0.05,
    seed: int = 0,
) -> list[float]:
    """Add Gaussian-like noise to a list of floats.

    Noise magnitude is proportional to each value:
    ``sigma = noise_fraction * abs(value)``.  When a value is exactly
    zero the noise is drawn from a small absolute-value normal
    distribution (``sigma = noise_fraction``).

    The Box-Muller transform is used to generate normally distributed
    noise without requiring NumPy or SciPy.

    Parameters
    ----------
    values:
        Source numeric values.
    noise_fraction:
        Fraction of the absolute value used as the noise standard
        deviation (default 0.05 → 5 %).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[float]
        Values with additive Gaussian noise applied.
    """
    import math

    rng = random.Random(seed)
    noisy: list[float] = []

    for v in values:
        sigma = noise_fraction * abs(v) if v != 0.0 else noise_fraction
        # Box-Muller transform
        u1 = rng.random() or 1e-10  # avoid log(0)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        noisy.append(v + sigma * z)

    return noisy


def bootstrap_sample(
    records: list[dict[str, Any]],
    n: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Draw *n* records with replacement from *records*.

    Parameters
    ----------
    records:
        Source dataset.
    n:
        Number of records to return.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Sampled records (shallow copies of the original dicts).
    """
    if not records:
        return []

    rng = random.Random(seed)
    return [dict(rng.choice(records)) for _ in range(n)]


# Fields that should receive noise during augmentation.
_NUMERIC_FIELDS = ("base_price", "demand", "competition_price", "revenue")


def augment_dataset(
    records: list[dict[str, Any]],
    multiplier: int = 2,
    noise_fraction: float = 0.05,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Expand a dataset by adding noisy copies of each record.

    The returned list contains the *original* records followed by
    ``multiplier - 1`` noisy copies.  Non-numeric fields are preserved
    unchanged in each copy.

    Parameters
    ----------
    records:
        Source dataset.
    multiplier:
        Total number of copies per original record including the
        original itself.  A value of 2 doubles the dataset.
    noise_fraction:
        Passed to :func:`add_noise` for each noisy copy.
    seed:
        Base random seed.  Each augmentation pass uses ``seed + pass``
        so copies are independently perturbed.

    Returns
    -------
    list[dict]
        Augmented dataset.
    """
    if multiplier < 1:
        return list(records)

    augmented = list(records)  # originals first

    for pass_idx in range(1, multiplier):
        pass_seed = seed + pass_idx
        for record in records:
            noisy_record = dict(record)
            for field in _NUMERIC_FIELDS:
                if field in record and record[field] is not None:
                    try:
                        original_val = float(record[field])
                        # add_noise on a single-element list
                        noisy_val = add_noise(
                            [original_val],
                            noise_fraction=noise_fraction,
                            seed=pass_seed,
                        )[0]
                        noisy_record[field] = noisy_val
                    except (TypeError, ValueError):
                        pass  # keep original if not numeric
            augmented.append(noisy_record)

    return augmented
