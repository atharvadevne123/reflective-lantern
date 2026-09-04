"""Structured, append-only audit log.

Provides an in-memory audit trail that records who did what and when,
with optional export to newline-delimited JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "AuditEntry",
    "AuditLog",
]


@dataclass(frozen=True)
class AuditEntry:
    """A single immutable audit record."""

    actor: str
    action: str
    resource: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"


class AuditLog:
    """Append-only in-memory audit log."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str = "success",
        **metadata: Any,
    ) -> AuditEntry:
        """Create and store an immutable audit entry, returning it to the caller.

        Args:
            actor: Identity of the entity performing the action (e.g. user ID).
            action: Verb describing what was done (e.g. "login", "delete").
            resource: Target of the action (e.g. "user:42", "model:v2").
            outcome: Result string, typically "success" or "failure".
            **metadata: Arbitrary extra fields attached to the entry.

        Returns:
            The newly created :class:`AuditEntry`.
        """
        entry = AuditEntry(
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            metadata=metadata,
        )
        self._entries.append(entry)
        return entry

    def search(
        self,
        actor: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        outcome: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[AuditEntry]:
        """Return entries matching all provided filter criteria (AND logic).

        Args:
            actor: Filter by exact actor value.
            action: Filter by exact action value.
            resource: Filter by exact resource value.
            outcome: Filter by exact outcome value.
            since: Lower-bound Unix timestamp (inclusive).
            until: Upper-bound Unix timestamp (inclusive).

        Returns:
            Filtered list of :class:`AuditEntry` objects in insertion order.
        """
        results = self._entries
        if actor:
            results = [e for e in results if e.actor == actor]
        if action:
            results = [e for e in results if e.action == action]
        if resource:
            results = [e for e in results if e.resource == resource]
        if outcome:
            results = [e for e in results if e.outcome == outcome]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        if until is not None:
            results = [e for e in results if e.timestamp <= until]
        return results

    def export_jsonl(self) -> str:
        """Return all entries serialised as newline-delimited JSON."""
        return "\n".join(json.dumps(asdict(e)) for e in self._entries)

    def __len__(self) -> int:
        """Return the total number of audit entries stored."""
        return len(self._entries)
