"""Standalone script to evaluate the current model on a holdout set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(n_holdout: int = 100, output_path: str | None = None) -> dict[str, Any]:
    """Evaluate the saved model on synthetic holdout data.

    Args:
        n_holdout: Number of holdout samples to generate.
        output_path: Optional path to write JSON metrics.

    Returns:
        Dict of evaluation metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from app.model import _to_dense, generate_synthetic_data, load_model

    result = load_model()
    if result is None:
        logger.error("No model found. Run training first.")
        sys.exit(1)

    pipeline, model = result
    samples, labels = generate_synthetic_data(n=n_holdout)

    X = pipeline.transform(samples)
    X_dense = _to_dense(X)
    proba = model.predict_proba(X_dense)[:, 1]
    preds = (proba >= 0.5).astype(int)
    y = np.array(labels)

    metrics: dict[str, Any] = {
        "n_holdout": n_holdout,
        "auc_roc": round(roc_auc_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, preds), 4),
        "f1": round(f1_score(y, preds, zero_division=0), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall": round(recall_score(y, preds, zero_division=0), 4),
        "viral_rate_true": round(float(y.mean()), 4),
        "viral_rate_pred": round(float(preds.mean()), 4),
    }

    logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))

    if output_path:
        Path(output_path).write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics written to %s", output_path)

    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the Social-Pulse model")
    parser.add_argument("-n", "--holdout", type=int, default=100)
    parser.add_argument("-o", "--output", default=None, help="JSON output path")
    args = parser.parse_args(argv)
    evaluate(n_holdout=args.holdout, output_path=args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
