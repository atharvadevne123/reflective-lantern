"""Simple file-based model registry for tracking trained model versions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path("/tmp/fleet_mind_model_registry.json")


def _load_registry() -> list[dict[str, Any]]:
    """Load the registry file or return an empty list.

    Returns:
        List of model entry dicts, most recent last.
    """
    if not _REGISTRY_PATH.exists():
        return []
    try:
        with open(_REGISTRY_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load model registry: %s", exc)
        return []


def _save_registry(entries: list[dict[str, Any]]) -> None:
    """Persist registry entries to disk.

    Args:
        entries: List of model entry dicts to write.
    """
    try:
        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REGISTRY_PATH, "w") as fh:
            json.dump(entries, fh, indent=2)
    except OSError as exc:
        logger.error("Failed to save model registry: %s", exc)


def register_model(version: str, metrics: dict[str, float], model_path: str) -> None:
    """Add a new model entry to the registry.

    Args:
        version: Semantic version string of the model.
        metrics: Training metrics (accuracy, roc_auc, etc.).
        model_path: Absolute path to the saved model file.
    """
    entries = _load_registry()
    entries.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "model_path": model_path,
    })
    _save_registry(entries)
    logger.info("Registered model version %s: AUC=%.4f", version, metrics.get("roc_auc", 0))


def list_models() -> list[dict[str, Any]]:
    """Return all registered model entries.

    Returns:
        List of model entry dicts sorted by training time (oldest first).
    """
    return _load_registry()


def latest_model_entry() -> dict[str, Any] | None:
    """Return the most recently registered model entry, or None.

    Returns:
        Most recent model entry dict, or None if registry is empty.
    """
    entries = _load_registry()
    return entries[-1] if entries else None
