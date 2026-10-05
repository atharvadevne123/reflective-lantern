"""High-level pipeline entry point for energy-domain predictions."""
from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["run_pipeline"]


def run_pipeline(df: pd.DataFrame) -> dict[str, Any]:
    """Train a model on *df* and return predictions for every row.

    Returns a dict with key ``predictions`` (list of floats) and ``metrics``.
    """
    from app.model import predict, train_model

    target_col = "consumption_kwh"
    if target_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{target_col}' column")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    bundle, metrics = train_model(X, y, save=False)
    preds = predict(bundle, X.values)
    return {
        "predictions": [float(p) for p in preds],
        "metrics": metrics,
    }
