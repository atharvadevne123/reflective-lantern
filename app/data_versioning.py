"""Dataset lineage and provenance tracking."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataSnapshot:
    """Immutable record of a dataset at a point in time.

    Attributes:
        name: Dataset name.
        version: Version string.
        source: URI or path where data originates.
        schema: Column name -> type mapping.
        row_count: Number of rows.
        checksum: SHA-256 hex digest of a canonical serialisation.
        tags: Arbitrary metadata key/value pairs.
        parent_versions: Versions this snapshot was derived from.
    """

    name: str
    version: str
    source: str
    schema: Dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    checksum: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    parent_versions: List[str] = field(default_factory=list)

    @classmethod
    def compute_checksum(cls, data: Any) -> str:
        """Compute a SHA-256 hex digest for arbitrary JSON-serialisable data.

        Args:
            data: Any JSON-serialisable object.

        Returns:
            Hex digest string.
        """
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


class DataLineage:
    """Tracks dataset snapshots and their lineage graph."""

    def __init__(self) -> None:
        """Initialise the lineage tracker with an empty snapshot store."""
        self._snapshots: Dict[str, List[DataSnapshot]] = {}

    def record(self, snapshot: DataSnapshot) -> None:
        """Record a new dataset snapshot.

        Args:
            snapshot: The snapshot to persist.

        Raises:
            ValueError: If the (name, version) pair already exists.
        """
        if snapshot.name not in self._snapshots:
            self._snapshots[snapshot.name] = []
        existing = {s.version for s in self._snapshots[snapshot.name]}
        if snapshot.version in existing:
            raise ValueError(
                f"Snapshot '{snapshot.name}' v{snapshot.version} already recorded"
            )
        self._snapshots[snapshot.name].append(snapshot)
        logger.info(
            "Recorded dataset '%s' v%s (%d rows)",
            snapshot.name, snapshot.version, snapshot.row_count,
        )

    def get(self, name: str, version: Optional[str] = None) -> Optional[DataSnapshot]:
        """Retrieve a snapshot by name and optional version.

        Args:
            name: Dataset name.
            version: Specific version, or None for the latest.

        Returns:
            Matching :class:`DataSnapshot` or None.
        """
        versions = self._snapshots.get(name)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for snap in versions:
            if snap.version == version:
                return snap
        return None

    def lineage(self, name: str, version: str) -> List[DataSnapshot]:
        """Return the ancestry chain of a snapshot (breadth-first).

        Args:
            name: Dataset name.
            version: Starting version.

        Returns:
            Ordered list of ancestor snapshots, excluding the start.
        """
        ancestors: List[DataSnapshot] = []
        queue = list(self.get(name, version).parent_versions if self.get(name, version) else [])
        seen: set = set()
        while queue:
            pv = queue.pop(0)
            if pv in seen:
                continue
            seen.add(pv)
            snap = self.get(name, pv)
            if snap:
                ancestors.append(snap)
                queue.extend(snap.parent_versions)
        return ancestors

    def list_versions(self, name: str) -> List[str]:
        """Return all recorded version strings for a dataset."""
        return [s.version for s in self._snapshots.get(name, [])]

    def list_datasets(self) -> List[str]:
        """Return all dataset names."""
        return list(self._snapshots.keys())


__all__ = ["DataLineage", "DataSnapshot"]
