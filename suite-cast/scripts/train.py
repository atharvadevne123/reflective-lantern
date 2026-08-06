"""CLI entry point to train the Suite-Cast ensemble from the command line.

Usage:
    python scripts/train.py --samples 3000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("suite_cast.train")


def main() -> int:
    """Train the ensemble and print resulting metrics.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Train the Suite-Cast demand model")
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic training samples (default 2000)",
    )
    args = parser.parse_args()

    from app.model import generate_sample_data, train_model

    logger.info("Generating %d training samples ...", args.samples)
    X, y = generate_sample_data(args.samples)

    logger.info("Training XGBoost + LightGBM ensemble with 5-fold CV ...")
    _, _, metrics = train_model(X, y)

    logger.info("Ensemble AUC: %.4f ± %.4f", metrics["auc_mean"], metrics["auc_std"])
    logger.info("Artifacts written: model.joblib, lgbm_model.joblib, metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
