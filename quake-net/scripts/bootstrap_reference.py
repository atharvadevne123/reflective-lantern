"""Seed the drift reference distribution from the training dataset.

Drift detection compares live traffic against a stored baseline. Without this
baseline every drift check silently no-ops, so run this once after training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.features import make_synthetic_dataset
from app.monitoring import REFERENCE_PATH, save_reference_distribution

NUMERIC_COLUMNS = (
    "latitude",
    "longitude",
    "depth_km",
    "station_count",
    "p_wave_amplitude",
    "s_wave_amplitude",
    "epicentral_distance_km",
)


def build_reference(n_samples: int = 2000, seed: int = 42) -> dict[str, list[float]]:
    """Build a per-feature reference distribution from a training sample."""
    df = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    reference = {col: [float(v) for v in df[col]] for col in NUMERIC_COLUMNS}
    reference["prediction"] = [float(v) for v in df["magnitude"]]
    return reference


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000, help="Rows to draw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=Path, default=None, help="Output path")
    args = parser.parse_args()

    reference = build_reference(n_samples=args.samples, seed=args.seed)

    if args.out is None:
        save_reference_distribution(reference)
        target = REFERENCE_PATH
    else:
        args.out.write_text(json.dumps(reference))
        target = args.out

    print(f"Reference distribution for {len(reference)} features written to {target}")


if __name__ == "__main__":
    main()
