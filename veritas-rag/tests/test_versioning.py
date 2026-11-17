"""Tests for veritas-rag document version store."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
