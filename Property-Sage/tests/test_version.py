"""Tests for version module."""
from app.version import VERSION, VERSION_INFO, get_version


def test_get_version():
    assert get_version() == "1.2.0"


def test_version_string():
    assert isinstance(VERSION, str)
    assert VERSION.count(".") == 2


def test_version_info():
    assert len(VERSION_INFO) == 3
    assert all(isinstance(x, int) for x in VERSION_INFO)


def test_version_info_matches_string():
    parts = [int(p) for p in VERSION.split(".")]
    assert list(VERSION_INFO) == parts
