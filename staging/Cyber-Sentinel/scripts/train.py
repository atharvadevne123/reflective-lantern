"""CLI script to train the Cyber-Sentinel ensemble model."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    """Entry point: train the model and print metrics."""
    logger.info("Starting Cyber-Sentinel model training")
    try:
        from app.model import train_model

        metrics = train_model()
        logger.info("Training complete:")
        logger.info("  F1 Score: %.4f", metrics["f1_score"])
        logger.info("  CV Mean:  %.4f", metrics["cv_mean"])
        logger.info("  CV Std:   %.4f", metrics["cv_std"])
        logger.info("  Samples:  %d", metrics["n_samples"])
        logger.info("  Estimators: %s", metrics["estimators"])
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
