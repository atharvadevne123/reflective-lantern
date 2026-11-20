"""Shadow mode — run a challenger model alongside production without affecting responses.

Shadow mode lets you evaluate a new model on live traffic without risk.
The challenger's predictions are logged but never returned to callers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_challenger_model: Any = None
_lock = threading.Lock()
_shadow_log: list[dict[str, Any]] = []
_MAX_LOG = 10_000


def register_challenger(model: Any) -> None:
    """Register a new challenger model for shadow-mode evaluation.

    Args:
        model: Trained model with a ``predict_proba`` method.
    """
    global _challenger_model
    with _lock:
        _challenger_model = model
    logger.info("Shadow mode: challenger model registered")


def shadow_predict(features: np.ndarray, production_result: dict[str, Any]) -> None:
    """Run the challenger model and log the comparison — never blocks the caller.

    Args:
        features: Feature array passed to the production model.
        production_result: The response already returned to the caller.
    """
    if _challenger_model is None:
        return

    def _run() -> None:
        try:
            prob = float(_challenger_model.predict_proba(features)[0, 1])
            challenger_maintenance = prob >= 0.5
            agreement = challenger_maintenance == production_result.get("maintenance_needed")
            entry = {
                "prod_score": production_result.get("anomaly_score"),
                "challenger_score": round(prob, 4),
                "agreement": agreement,
            }
            with _lock:
                _shadow_log.append(entry)
                if len(_shadow_log) > _MAX_LOG:
                    _shadow_log.pop(0)
            logger.debug("Shadow prediction logged: agreement=%s", agreement)
        except Exception as exc:
            logger.warning("Shadow prediction failed silently: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


def get_shadow_stats() -> dict[str, Any]:
    """Return aggregate agreement statistics between production and challenger.

    Returns:
        Dict with 'total_compared', 'agreement_rate', 'challenger_available',
        'mean_score_delta', and 'max_score_delta'.
    """
    with _lock:
        log_copy = list(_shadow_log)

    if not log_copy:
        return {
            "total_compared": 0,
            "agreement_rate": None,
            "challenger_available": _challenger_model is not None,
            "mean_score_delta": None,
            "max_score_delta": None,
        }

    agreement_rate = sum(1 for e in log_copy if e["agreement"]) / len(log_copy)
    deltas = [
        abs(e["challenger_score"] - e["prod_score"])
        for e in log_copy
        if e["prod_score"] is not None and e["challenger_score"] is not None
    ]
    return {
        "total_compared": len(log_copy),
        "agreement_rate": round(agreement_rate, 4),
        "challenger_available": _challenger_model is not None,
        "mean_score_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "max_score_delta": round(max(deltas), 4) if deltas else None,
    }


def clear_shadow_log() -> None:
    """Clear the shadow comparison log (useful during tests or model switches)."""
    with _lock:
        _shadow_log.clear()
    logger.info("Shadow mode: comparison log cleared")
