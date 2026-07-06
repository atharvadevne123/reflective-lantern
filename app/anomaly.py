"""Anomaly detection for property valuations.

Flags properties whose predicted value deviates significantly from the
neighbourhood median using Z-score and IQR methods. This module complements
the KS-test drift monitor by operating on individual predictions rather than
feature distributions.

Typical usage
-------------
from app.anomaly import detect_valuation_anomaly
result = detect_valuation_anomaly(predicted=250_000, neighborhood_median=600_000)
if result["is_anomaly"]:
    print("Possibly underpriced or data entry error")
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER = 1.5


def detect_valuation_anomaly(
    predicted: float,
    neighborhood_median: float,
    neighborhood_std: float | None = None,
    reference_values: list[float] | None = None,
) -> dict[str, Any]:
    """Detect whether a predicted valuation is anomalous.

    Uses Z-score if a standard deviation is provided, otherwise falls back to
    a simple ratio-based check against the neighbourhood median.

    Args:
        predicted: The model's predicted property value.
        neighborhood_median: Median value for comparable properties in the area.
        neighborhood_std: Standard deviation of comparable values (optional).
        reference_values: Full list of reference values for IQR computation.

    Returns:
        Dict with keys: is_anomaly (bool), method (str), score (float),
        deviation_pct (float), direction ('low'|'high'|'normal').
    """
    if predicted <= 0 or neighborhood_median <= 0:
        return {"is_anomaly": False, "method": "skipped", "score": 0.0,
                "deviation_pct": 0.0, "direction": "normal"}

    deviation_pct = (predicted - neighborhood_median) / neighborhood_median * 100

    if neighborhood_std and neighborhood_std > 0:
        zscore = abs(predicted - neighborhood_median) / neighborhood_std
        is_anomaly = zscore > ZSCORE_THRESHOLD
        method = "zscore"
        score = float(zscore)
    elif reference_values and len(reference_values) >= 4:
        arr = np.array(reference_values, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        is_anomaly = predicted < lower or predicted > upper
        score = float(max(0, predicted - upper) / (iqr + 1e-9) if predicted > upper
                      else max(0, lower - predicted) / (iqr + 1e-9))
        method = "iqr"
    else:
        ratio = predicted / neighborhood_median
        is_anomaly = ratio <= 0.4 or ratio > 2.5
        score = float(abs(ratio - 1.0))
        method = "ratio"

    direction = "low" if predicted < neighborhood_median else ("high" if predicted > neighborhood_median else "normal")

    if is_anomaly:
        logger.warning(
            "Valuation anomaly detected: predicted=%.0f median=%.0f deviation=%.1f%% method=%s",
            predicted, neighborhood_median, deviation_pct, method,
        )

    return {
        "is_anomaly": bool(is_anomaly),
        "method": method,
        "score": round(score, 4),
        "deviation_pct": round(deviation_pct, 2),
        "direction": direction,
    }
