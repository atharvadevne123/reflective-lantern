"""Versioned in-memory feature store for ML training and serving."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """A named, versioned collection of features.

    Attributes:
        name: Feature set identifier.
        version: Semantic version string.
        features: Dict mapping feature name to value.
        description: Optional human-readable description.
    """

    name: str
    version: str
    features: Dict[str, Any]
    description: str = ""

    def get(self, feature: str, default: Any = None) -> Any:
        """Retrieve a feature value by name."""
        return self.features.get(feature, default)

    def keys(self) -> List[str]:
        """Return all feature names."""
        return list(self.features.keys())


class FeatureStore:
    """Manages multiple versioned feature sets.

    Supports multiple versions per feature set name, retrieving the latest
    or a specific version by SemVer string.
    """

    def __init__(self) -> None:
        """Initialise the feature store with an empty version-history dictionary."""
        self._store: Dict[str, List[FeatureSet]] = {}

    def publish(self, feature_set: FeatureSet) -> None:
        """Publish a feature set, appending to its version history.

        Args:
            feature_set: The feature set to publish.
        """
        if feature_set.name not in self._store:
            self._store[feature_set.name] = []
        versions = self._store[feature_set.name]
        existing_versions = {fs.version for fs in versions}
        if feature_set.version in existing_versions:
            raise ValueError(
                f"Version '{feature_set.version}' already exists for '{feature_set.name}'"
            )
        versions.append(feature_set)
        logger.info(
            "Published feature set '%s' v%s", feature_set.name, feature_set.version
        )

    def get_latest(self, name: str) -> Optional[FeatureSet]:
        """Return the most recently published version of a feature set.

        Args:
            name: Feature set name.

        Returns:
            Latest :class:`FeatureSet`, or None if not found.
        """
        versions = self._store.get(name)
        if not versions:
            logger.warning("Feature set '%s' not found", name)
            return None
        return versions[-1]

    def get_version(self, name: str, version: str) -> Optional[FeatureSet]:
        """Return a specific version of a feature set.

        Args:
            name: Feature set name.
            version: Version string.

        Returns:
            Matching :class:`FeatureSet`, or None.
        """
        for fs in self._store.get(name, []):
            if fs.version == version:
                return fs
        return None

    def list_versions(self, name: str) -> List[str]:
        """Return all published version strings for a feature set name."""
        return [fs.version for fs in self._store.get(name, [])]

    def list_names(self) -> List[str]:
        """Return all registered feature set names."""
        return list(self._store.keys())

    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a feature set or a specific version.

        Args:
            name: Feature set name.
            version: Specific version to delete, or None to delete all.

        Returns:
            True if something was deleted.
        """
        if name not in self._store:
            return False
        if version is None:
            del self._store[name]
            return True
        before = len(self._store[name])
        self._store[name] = [fs for fs in self._store[name] if fs.version != version]
        if not self._store[name]:
            del self._store[name]
        return len(self._store.get(name, [])) < before


__all__ = ["FeatureSet", "FeatureStore"]
