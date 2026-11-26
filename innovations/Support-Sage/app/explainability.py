"""SHAP-based model explainability for Support-Sage predictions."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def explain_prediction(model, X, feature_names: list[str] | None = None) -> dict[str, Any]:
    """Return SHAP values for a single prediction, if shap is installed."""
    if not _SHAP_AVAILABLE:
        return {"available": False, "reason": "shap not installed"}
    try:
        explainer = shap.TreeExplainer(model.named_steps["clf"].estimators_[0])
        features = model.named_steps["features"].transform(X)
        shap_values = explainer.shap_values(features)
        return {"available": True, "shap_values_shape": str(shap_values[0].shape)}
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return {"available": False, "reason": str(exc)}
