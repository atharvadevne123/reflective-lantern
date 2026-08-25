"""Versioned model registry for tracking ML model artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Lifecycle stage of a registered model version."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """A single model version entry.

    Attributes:
        name: Model name.
        version: Semantic version string.
        artifact_path: Path or URI to the serialised model file.
        stage: Current lifecycle stage.
        metrics: Training / evaluation metrics at registration time.
        tags: Arbitrary key-value tags.
    """

    name: str
    version: str
    artifact_path: str
    stage: ModelStage = ModelStage.STAGING
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """Central registry for ML model versions.

    Supports registering, staging, promoting to production, archiving,
    and querying model versions.
    """

    def __init__(self) -> None:
        self._models: dict[str, list[ModelVersion]] = {}

    def register(self, model: ModelVersion) -> None:
        """Register a new model version.

        Args:
            model: The model version to register.

        Raises:
            ValueError: If the version already exists for this model name.
        """
        if model.name not in self._models:
            self._models[model.name] = []
        existing = {mv.version for mv in self._models[model.name]}
        if model.version in existing:
            raise ValueError(f"Version '{model.version}' already registered for model '{model.name}'")
        self._models[model.name].append(model)
        logger.info("Registered model '%s' v%s", model.name, model.version)

    def transition_stage(self, name: str, version: str, stage: ModelStage) -> bool:
        """Transition a model version to a new stage.

        Args:
            name: Model name.
            version: Version string.
            stage: Target stage.

        Returns:
            True if found and transitioned, False otherwise.
        """
        mv = self._get(name, version)
        if mv is None:
            return False
        mv.stage = stage
        logger.info("Model '%s' v%s -> %s", name, version, stage.value)
        return True

    def get_production(self, name: str) -> ModelVersion | None:
        """Return the most recent PRODUCTION version of a model."""
        prod = [mv for mv in self._models.get(name, []) if mv.stage is ModelStage.PRODUCTION]
        return prod[-1] if prod else None

    def get_latest(self, name: str) -> ModelVersion | None:
        """Return the most recently registered version."""
        versions = self._models.get(name)
        return versions[-1] if versions else None

    def _get(self, name: str, version: str) -> ModelVersion | None:
        for mv in self._models.get(name, []):
            if mv.version == version:
                return mv
        return None

    def list_versions(self, name: str) -> list[ModelVersion]:
        """Return all versions of a model."""
        return list(self._models.get(name, []))

    def list_models(self) -> list[str]:
        """Return names of all registered models."""
        return list(self._models.keys())

    def add_tag(self, name: str, version: str, key: str, value: str) -> bool:
        """Add a tag to a model version."""
        mv = self._get(name, version)
        if mv is None:
            return False
        mv.tags[key] = value
        return True


__all__ = ["ModelRegistry", "ModelStage", "ModelVersion"]
