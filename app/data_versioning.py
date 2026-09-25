"""Dataset lineage and provenance tracking."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    source: str = ""
    schema: dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    checksum: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    parent_versions: list[str] = field(default_factory=list)

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
        """Initialise an empty lineage store."""
        self._snapshots: dict[str, list[DataSnapshot]] = {}

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
            raise ValueError(f"Snapshot '{snapshot.name}' v{snapshot.version} already recorded")
        self._snapshots[snapshot.name].append(snapshot)
        logger.info(
            "Recorded dataset '%s' v%s (%d rows)",
            snapshot.name,
            snapshot.version,
            snapshot.row_count,
        )

    def get(self, name: str, version: str | None = None) -> DataSnapshot | None:
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

    def lineage(self, name: str, version: str) -> list[DataSnapshot]:
        """Return the ancestry chain of a snapshot (breadth-first).

        Args:
            name: Dataset name.
            version: Starting version.

        Returns:
            Ordered list of ancestor snapshots, excluding the start.
        """
        ancestors: list[DataSnapshot] = []
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

    def list_versions(self, name: str) -> list[str]:
        """Return all recorded version strings for a dataset."""
        return [s.version for s in self._snapshots.get(name, [])]

    def list_datasets(self) -> list[str]:
        """Return all dataset names."""
        return list(self._snapshots.keys())

    def snapshot_count(self, name: str) -> int:
        """Return the number of recorded snapshots for a dataset."""
        return len(self._snapshots.get(name, []))

    def total_rows(self, name: str) -> int:
        """Return the sum of row_count across all snapshots for a dataset."""
        return sum(s.row_count for s in self._snapshots.get(name, []))

    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete a snapshot or all snapshots for a dataset. Returns True if something was removed."""
        if name not in self._snapshots:
            return False
        if version is None:
            del self._snapshots[name]
            return True
        before = len(self._snapshots[name])
        self._snapshots[name] = [s for s in self._snapshots[name] if s.version != version]
        if not self._snapshots[name]:
            del self._snapshots[name]
        return len(self._snapshots.get(name, [])) < before


def export_lineage_json(lineage: DataLineage, path: Path | str) -> Path:
    """Serialise *lineage* to a JSON file at *path*.

    Parent directories are created automatically via
    :func:`pathlib.Path.mkdir`.  The file is written with UTF-8 encoding and
    a two-space indent for readability.

    Args:
        lineage: The :class:`DataLineage` registry to export.
        path: Destination file path (string or :class:`~pathlib.Path`).

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict]] = {}
    for dataset in lineage.list_datasets():
        payload[dataset] = [
            {
                "version": snap.version,
                "row_count": snap.row_count,
                "source": snap.source,
                "checksum": snap.checksum,
                "created_at": snap.created_at,
                "metadata": snap.metadata,
            }
            for snap in lineage._snapshots[dataset]
        ]
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


__all__ = ["DataLineage", "DataSnapshot", "export_lineage_json"]
