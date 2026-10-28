"""Data quality checks for incoming fleet telemetry batches."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.features import RAW_FEATURES

logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return count of missing values per feature column.

    Args:
        df: DataFrame of telemetry readings with RAW_FEATURES columns.

    Returns:
        Dict mapping feature name to missing count (0 means no missing).
    """
    return {col: int(df[col].isna().sum()) for col in RAW_FEATURES if col in df.columns}


def check_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 4.0,
) -> dict[str, list[int]]:
    """Return row indices with z-score outliers per feature.

    Args:
        df: DataFrame of telemetry readings.
        threshold: Z-score magnitude above which a value is considered an outlier.

    Returns:
        Dict mapping feature name to list of outlier row indices.
    """
    outliers: dict[str, list[int]] = {}
    for col in RAW_FEATURES:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) < 5:
            continue
        mean = values.mean()
        std = values.std()
        if std == 0:
            continue
        z_scores = ((df[col] - mean) / std).abs()
        indices = z_scores[z_scores > threshold].index.tolist()
        if indices:
            outliers[col] = [int(i) for i in indices]
    return outliers


def assess_batch_quality(readings: list[dict[str, Any]]) -> dict[str, Any]:
    """Run a full data quality assessment on a batch of telemetry dicts.

    Args:
        readings: List of raw telemetry dicts (vehicle_id may be present but is ignored).

    Returns:
        Dict with 'total_rows', 'missing_values', 'outliers', and 'quality_score'
        (float 0–1 where 1.0 means perfect quality).
    """
    if not readings:
        return {"total_rows": 0, "quality_score": 0.0}

    numeric_readings = [
        {k: v for k, v in r.items() if k in RAW_FEATURES}
        for r in readings
    ]
    df = pd.DataFrame(numeric_readings)

    missing = check_missing_values(df)
    outliers = check_outliers_zscore(df)

    total_cells = len(df) * len(RAW_FEATURES)
    missing_cells = sum(missing.values())
    outlier_cells = sum(len(v) for v in outliers.values())

    quality_score = max(0.0, 1.0 - (missing_cells + outlier_cells * 0.5) / max(total_cells, 1))

    result = {
        "total_rows": len(df),
        "missing_values": missing,
        "outliers": {k: len(v) for k, v in outliers.items()},
        "quality_score": round(quality_score, 4),
    }
    logger.info("Batch quality assessment: score=%.4f missing=%d outliers=%d",
                quality_score, missing_cells, outlier_cells)
    return result
