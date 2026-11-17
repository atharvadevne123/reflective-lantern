"""Tests for Energy-Seer OpenAPI metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.openapi_meta import TAGS_METADATA


def test_tags_metadata_is_list():
    assert isinstance(TAGS_METADATA, list)


def test_tags_metadata_not_empty():
    assert len(TAGS_METADATA) > 0


def test_each_tag_has_name():
    for tag in TAGS_METADATA:
        assert "name" in tag
        assert isinstance(tag["name"], str)
        assert tag["name"]


def test_each_tag_has_description():
    for tag in TAGS_METADATA:
        assert "description" in tag
        assert isinstance(tag["description"], str)


def test_tag_names_are_unique():
    names = [t["name"] for t in TAGS_METADATA]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("expected_tag", ["Forecast", "Anomaly Detection", "Monitoring", "System"])
def test_expected_tags_present(expected_tag):
    names = [t["name"] for t in TAGS_METADATA]
    assert expected_tag in names
