"""Simple model registry for tracking deployed model versions."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(os.environ.get("MODEL_REGISTRY_PATH", "/tmp/signal_forge_registry.json"))


@dataclass
class ModelRecord:
    """Metadata for one registered model version."""

    version: str
    path: str
    trained_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return asdict(self)


class ModelRegistry:
    """File-backed registry of trained model versions."""

    def __init__(self, registry_path: Path = REGISTRY_PATH) -> None:
        self._path = registry_path
        self._records: list[ModelRecord] = self._load()

    def _load(self) -> list[ModelRecord]:
        """Load records from JSON file, returning empty list if absent."""
        if not self._path.exists():
            return []
        try:
            with open(self._path) as f:
                raw = json.load(f)
            return [ModelRecord(**r) for r in raw]
        except Exception as e:
            logger.warning("Failed to load registry at %s: %s", self._path, e)
            return []

    def _save(self) -> None:
        """Persist records to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)

    def register(
        self,
        version: str,
        path: str,
        metrics: dict[str, float] | None = None,
        set_active: bool = True,
    ) -> ModelRecord:
        """Register a new model version, optionally marking it active.

        Args:
            version: Semantic version string (e.g. "1.0.0").
            path: File-system path to the serialised model.
            metrics: Optional evaluation metrics dict.
            set_active: If True, deactivate all other versions first.

        Returns:
            The newly created ModelRecord.
        """
        if set_active:
            for r in self._records:
                r.active = False
        record = ModelRecord(
            version=version,
            path=path,
            trained_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            active=set_active,
        )
        self._records.append(record)
        self._save()
        logger.info("Model registered: version=%s active=%s", version, set_active)
        return record

    def get_active(self) -> ModelRecord | None:
        """Return the currently active model record, or None."""
        for r in reversed(self._records):
            if r.active:
                return r
        return None

    def list_all(self) -> list[ModelRecord]:
        """Return all registered model records, newest first."""
        return list(reversed(self._records))
