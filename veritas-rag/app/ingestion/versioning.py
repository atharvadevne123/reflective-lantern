"""Document version chains.

Documents are identified by their *source identity* (source name). Re-ingesting
a source with changed content creates a new version rather than a new
document, so queries always retrieve the latest content while older versions
remain addressable for audit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..schemas import Document


def make_doc_id(source: str, version: int) -> str:
    """Stable doc id derived from source identity and version."""
    stem = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"doc-{stem}-v{version}"


@dataclass
class VersionChain:
    """All versions of one logical document, newest last."""

    source: str
    versions: list[Document] = field(default_factory=list)

    @property
    def latest(self) -> Document:
        """Return the most recently added version of this document."""
        return self.versions[-1]


class VersionStore:
    """Tracks version chains keyed by source identity."""

    def __init__(self) -> None:
        """Initialise the store with an empty chain registry."""
        self._chains: dict[str, VersionChain] = {}

    def next_version(self, source: str) -> int:
        """Version number the next ingest of this source should carry."""
        chain = self._chains.get(source)
        return 1 if chain is None else chain.latest.version + 1

    def is_unchanged(self, source: str, new_content_hash: str) -> bool:
        """True when the source's latest version has identical content."""
        chain = self._chains.get(source)
        return chain is not None and chain.latest.content_hash == new_content_hash

    def add(self, doc: Document) -> None:
        """Append a document to its source's version chain."""
        chain = self._chains.setdefault(doc.source, VersionChain(source=doc.source))
        chain.versions.append(doc)

    def latest(self, source: str) -> Document | None:
        """Return the most recent document for *source*, or None if unknown."""
        chain = self._chains.get(source)
        return chain.latest if chain else None

    def history(self, source: str) -> list[Document]:
        """Return all versions of *source* in chronological order."""
        chain = self._chains.get(source)
        return list(chain.versions) if chain else []

    def superseded_doc_ids(self, source: str) -> list[str]:
        """Doc ids of all non-latest versions (to be dropped from indexes)."""
        chain = self._chains.get(source)
        if chain is None or len(chain.versions) < 2:
            return []
        return [d.doc_id for d in chain.versions[:-1]]

    def __len__(self) -> int:
        """Return the number of distinct sources tracked by this store."""
        return len(self._chains)
