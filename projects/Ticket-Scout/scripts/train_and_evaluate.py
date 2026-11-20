"""Standalone training and evaluation script for Ticket-Scout."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Ticket-Scout models")
    parser.add_argument("--n-samples", type=int, default=3000, help="Training set size")
    parser.add_argument("--model-dir", type=str, default="./models", help="Model output directory")
    parser.add_argument("--output-json", type=str, default=None, help="Write metrics JSON to file")
    args = parser.parse_args()

    import os
    os.environ["MODEL_DIR"] = args.model_dir

    from app.model import generate_synthetic_data, train_models

    logger.info("Generating %d training samples", args.n_samples)
    df = generate_synthetic_data(n_samples=args.n_samples)
    metrics = train_models(df)

    print(json.dumps(metrics, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics written to %s", args.output_json)

    # Exit 1 if quality below threshold
    if metrics.get("breach_auc_mean", 0) < 0.60:
        logger.error("SLA breach AUC too low: %.3f", metrics["breach_auc_mean"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
