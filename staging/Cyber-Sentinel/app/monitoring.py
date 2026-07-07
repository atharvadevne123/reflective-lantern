"""KS-test drift detection and prediction health monitoring."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05
_reference_data: dict[str, np.ndarray] = {}
_prediction_log: list[dict[str, Any]] = []


def set_reference_distribution(feature_name: str, values: np.ndarray) -> None:
    """Store the reference distribution for a feature.

    Args:
        feature_name: Identifier for the feature.
        values: Array of reference (training) values.
    """
    _reference_data[feature_name] = np.asarray(values, dtype=float)
    logger.info("Reference distribution set for feature '%s' (%d samples)", feature_name, len(values))


def compute_drift(feature_name: str, current_values: np.ndarray) -> dict[str, Any]:
    """Run a two-sample KS test between reference and current distributions.

    Args:
        feature_name: Name of the feature to test.
        current_values: Array of values from the current window.

    Returns:
        Dict with ks_statistic, p_value, drift_detected, and feature_name.
    """
    if feature_name not in _reference_data:
        logger.warning("No reference distribution for feature '%s'", feature_name)
        return {
            "feature_name": feature_name,
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "error": "no_reference",
        }

    reference = _reference_data[feature_name]
    current = np.asarray(current_values, dtype=float)

    if len(current) < 2:
        return {
            "feature_name": feature_name,
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "error": "insufficient_samples",
        }

    try:
        result = stats.ks_2samp(reference, current)
        drift_detected = bool(result.pvalue < DRIFT_THRESHOLD)
        if drift_detected:
            logger.warning(
                "Drift detected | feature=%s ks=%.4f p=%.4f n_ref=%d n_cur=%d",
                feature_name,
                result.statistic,
                result.pvalue,
                len(reference),
                len(current),
            )
        return {
            "feature_name": feature_name,
            "ks_statistic": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 6),
            "drift_detected": drift_detected,
        }
    except Exception as exc:
        logger.error("KS test failed for '%s': %s", feature_name, exc)
        return {
            "feature_name": feature_name,
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "error": str(exc),
        }


def log_prediction(prediction: dict[str, Any], features: list[float]) -> None:
    """Append a prediction to the in-memory log for drift analysis."""
    _prediction_log.append({"prediction": prediction, "features": features})
    if len(_prediction_log) > 10000:
        _prediction_log.pop(0)


def prediction_health() -> dict[str, Any]:
    """Compute summary statistics over the prediction log.

    Returns:
        Dict with total predictions, attack rate, average confidence.
    """
    if not _prediction_log:
        return {"total_predictions": 0, "attack_rate": 0.0, "avg_confidence": 0.0}

    total = len(_prediction_log)
    attacks = sum(1 for p in _prediction_log if p["prediction"].get("label") == "attack")
    confidences = [p["prediction"].get("confidence", 0.0) for p in _prediction_log]

    return {
        "total_predictions": total,
        "attack_rate": round(attacks / total, 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "max_confidence": round(float(np.max(confidences)), 4),
    }


def check_alerts(drift_results: list[dict[str, Any]]) -> list[str]:
    """Return a list of alert messages for any detected drifts.

    Args:
        drift_results: List of results from compute_drift().

    Returns:
        List of human-readable alert strings.
    """
    alerts: list[str] = []
    for result in drift_results:
        if result.get("drift_detected"):
            alerts.append(
                f"DRIFT ALERT: feature '{result['feature_name']}' "
                f"KS={result['ks_statistic']:.4f} p={result['p_value']:.4f}"
            )
    return alerts


def get_feature_drift_summary(feature_indices: list[int] | None = None) -> dict[str, Any]:
    """Compute drift for recent predictions against reference data.

    Args:
        feature_indices: Which feature indices to check (default: all available).

    Returns:
        Dict mapping feature names to their drift results.
    """
    if not _prediction_log:
        return {"error": "no_predictions_logged"}

    feature_matrix = np.array([p["features"] for p in _prediction_log])
    n_features = feature_matrix.shape[1] if feature_matrix.ndim == 2 else 0
    indices = feature_indices or list(range(min(n_features, len(_reference_data))))

    results: dict[str, Any] = {}
    for idx in indices:
        name = f"feature_{idx}"
        if idx < n_features:
            results[name] = compute_drift(name, feature_matrix[:, idx])
    return results


def reset_monitoring() -> None:
    """Clear all reference data and prediction logs (for testing)."""
    _reference_data.clear()
    _prediction_log.clear()
