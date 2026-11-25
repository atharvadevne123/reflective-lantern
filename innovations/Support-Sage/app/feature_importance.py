"""Feature importance extraction from the trained ensemble."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def get_feature_importance(top_n: int = 20) -> dict[str, Any]:
    """Return top-N feature importances from the RandomForest estimator."""
    model_path = Path("models/priority_model.joblib")
    if not model_path.exists():
        return {"error": "Model not found - run /api/v1/train first"}
    pipe = joblib.load(model_path)
    try:
        rf = pipe.named_steps["clf"].estimators_[2][1]  # RandomForest
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return {
            "top_features": [
                {"rank": int(i + 1), "index": int(idx), "importance": round(float(importances[idx]), 6)}
                for i, idx in enumerate(indices)
            ],
            "total_features": int(len(importances)),
        }
    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)
        return {"error": str(exc)}
