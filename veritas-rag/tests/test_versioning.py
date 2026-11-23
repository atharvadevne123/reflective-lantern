"""Tests for veritas-rag document version store."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.versioning import VersionStore, make_doc_id
from app.schemas import Document


def _make_doc(source="src.txt", version=1, content_hash="abc123", text="hello"):
    return Document(
        doc_id=make_doc_id(source, version),
        source=source,
        source_type="text",
        text=text,
        pages=[text],
        created_at="2026-01-01T00:00:00",
        version=version,
        content_hash=content_hash,
    )


def test_make_doc_id_returns_string():
    doc_id = make_doc_id("source.txt", 1)
    assert isinstance(doc_id, str)


def test_make_doc_id_deterministic():
    assert make_doc_id("src.txt", 1) == make_doc_id("src.txt", 1)


def test_make_doc_id_different_sources():
    assert make_doc_id("a.txt", 1) != make_doc_id("b.txt", 1)


def test_make_doc_id_different_versions():
    assert make_doc_id("src.txt", 1) != make_doc_id("src.txt", 2)


def test_next_version_new_source():
    store = VersionStore()
    assert store.next_version("new.txt") == 1


def test_next_version_after_first_add():
    store = VersionStore()
    doc = _make_doc(source="f.txt", version=1)
    store.add(doc)
    assert store.next_version("f.txt") == 2


def test_is_unchanged_no_prior_version():
    store = VersionStore()
    assert not store.is_unchanged("new.txt", "hash123")


def test_is_unchanged_same_hash():
    store = VersionStore()
    doc = _make_doc(content_hash="abc")
    store.add(doc)
    assert store.is_unchanged("src.txt", "abc")


def test_is_unchanged_different_hash():
    store = VersionStore()
    doc = _make_doc(content_hash="abc")
    store.add(doc)
    assert not store.is_unchanged("src.txt", "xyz")


def test_add_multiple_versions():
    store = VersionStore()
    doc1 = _make_doc(version=1, content_hash="h1")
    doc2 = _make_doc(version=2, content_hash="h2")
    store.add(doc1)
    store.add(doc2)
    assert store.next_version("src.txt") == 3


def test_latest_returns_last_added():
    store = VersionStore()
    doc1 = _make_doc(version=1, content_hash="h1")
    doc2 = _make_doc(version=2, content_hash="h2")
    store.add(doc1)
    store.add(doc2)
    chain = store._chains["src.txt"]
    assert chain.latest.version == 2


def test_latest_on_unknown_source_returns_none():
    store = VersionStore()
    assert store.latest("nonexistent.txt") is None


def test_history_on_unknown_source_returns_empty_list():
    store = VersionStore()
    assert store.history("nonexistent.txt") == []


def test_history_returns_all_versions_in_order():
    store = VersionStore()
    for v in range(1, 5):
        store.add(_make_doc(version=v, content_hash=f"h{v}"))
    history = store.history("src.txt")
    assert [d.version for d in history] == [1, 2, 3, 4]


def test_superseded_doc_ids_no_prior():
    store = VersionStore()
    doc = _make_doc(version=1)
    store.add(doc)
    assert store.superseded_doc_ids("src.txt") == []


def test_superseded_doc_ids_after_two_versions():
    store = VersionStore()
    doc1 = _make_doc(version=1, content_hash="h1")
    doc2 = _make_doc(version=2, content_hash="h2")
    store.add(doc1)
    store.add(doc2)
    superseded = store.superseded_doc_ids("src.txt")
    assert len(superseded) == 1
    assert doc1.doc_id in superseded


def test_len_counts_distinct_sources():
    store = VersionStore()
    store.add(_make_doc(source="a.txt", version=1, content_hash="h1"))
    store.add(_make_doc(source="b.txt", version=1, content_hash="h2"))
    store.add(_make_doc(source="a.txt", version=2, content_hash="h3"))
    assert len(store) == 2


def test_is_unchanged_new_source_returns_false():
    store = VersionStore()
    assert store.is_unchanged("new.txt", "hash123") is False


def test_is_unchanged_same_hash_returns_true():
    store = VersionStore()
    store.add(_make_doc(source="file.txt", version=1, content_hash="abc123"))
    assert store.is_unchanged("file.txt", "abc123") is True


def test_is_unchanged_different_hash_returns_false():
    store = VersionStore()
    store.add(_make_doc(source="file.txt", version=1, content_hash="abc123"))
    assert store.is_unchanged("file.txt", "def456") is False


def test_next_version_new_source_is_one():
    store = VersionStore()
    assert store.next_version("brand-new.txt") == 1


def test_history_length_matches_add_count():
    store = VersionStore()
    for v in range(1, 6):
        store.add(_make_doc(version=v, content_hash=f"hash{v}"))
    assert len(store.history("src.txt")) == 5


def test_make_doc_id_stable():
    from app.ingestion.versioning import make_doc_id
    id1 = make_doc_id("file.txt", 1)
    id2 = make_doc_id("file.txt", 1)
    assert id1 == id2


def test_make_doc_id_different_versions():
    from app.ingestion.versioning import make_doc_id
    id1 = make_doc_id("file.txt", 1)
    id2 = make_doc_id("file.txt", 2)
    assert id1 != id2
