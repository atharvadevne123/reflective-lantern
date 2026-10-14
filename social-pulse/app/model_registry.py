"""Simple file-based model registry for tracking model versions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("/tmp/social_pulse_models/registry.json")


def _load_registry() -> list[dict[str, Any]]:
    """Load the registry JSON, returning an empty list if missing."""
    if not REGISTRY_PATH.exists():
        return []
    try:
        with open(REGISTRY_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Registry load failed: %s", exc)
        return []


def _save_registry(entries: list[dict[str, Any]]) -> None:
    """Persist *entries* to the registry JSON file."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w") as fh:
            json.dump(entries, fh, indent=2)
    except OSError as exc:
        logger.error("Registry save failed: %s", exc)


def register_model(
    run_id: str,
    model_version: str,
    auc_roc: float,
    f1: float,
    train_samples: int,
    promoted: bool = False,
) -> dict[str, Any]:
    """Add a new model entry to the registry.

    Args:
        run_id: Unique run identifier from TrainResult.
        model_version: Semantic version string.
        auc_roc: AUC-ROC on the training run's validation set.
        f1: F1 score on the training run's validation set.
        train_samples: Number of training samples used.
        promoted: Whether this model is currently serving production traffic.

    Returns:
        The newly created registry entry dict.
    """
    entry: dict[str, Any] = {
        "run_id": run_id,
        "model_version": model_version,
        "auc_roc": round(auc_roc, 4),
        "f1": round(f1, 4),
        "train_samples": train_samples,
        "promoted": promoted,
        "registered_at": datetime.utcnow().isoformat(),
    }
    entries = _load_registry()
    entries.append(entry)
    _save_registry(entries)
    logger.info("Registered model run_id=%s version=%s AUC=%.4f", run_id, model_version, auc_roc)
    return entry


def list_models() -> list[dict[str, Any]]:
    """Return all registered models, newest first.

    Returns:
        List of model entry dicts sorted by registration time descending.
    """
    return list(reversed(_load_registry()))


def get_latest_promoted() -> dict[str, Any] | None:
    """Return the most recently promoted model entry, or None.

    Returns:
        The promoted model dict, or None if no model is promoted.
    """
    for entry in reversed(_load_registry()):
        if entry.get("promoted"):
            return entry
    return None


def promote_model(run_id: str) -> bool:
    """Mark the model with *run_id* as promoted and demote all others.

    Args:
        run_id: The run ID of the model to promote.

    Returns:
        True if the model was found and promoted, False otherwise.
    """
    entries = _load_registry()
    found = False
    for entry in entries:
        if entry["run_id"] == run_id:
            entry["promoted"] = True
            found = True
        else:
            entry["promoted"] = False
    if found:
        _save_registry(entries)
        logger.info("Promoted model run_id=%s", run_id)
    else:
        logger.warning("promote_model: run_id=%s not found", run_id)
    return found
