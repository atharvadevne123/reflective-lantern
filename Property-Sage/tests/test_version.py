"""Tests for version module."""

import pytest

from app.version import get_api_version, get_version


def test_version_format():
    v = get_version()
    parts = v.split(".")
    assert len(parts) == 3


def test_api_version_prefix():
    assert get_api_version().startswith("v")


@pytest.mark.parametrize("func", [get_version, get_api_version])
def test_returns_string(func):
    assert isinstance(func(), str)
