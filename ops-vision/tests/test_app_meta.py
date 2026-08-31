"""Tests for Ops-Vision package metadata."""

import app


class TestPackageMetadata:
    """Tests that package-level metadata fields are correct."""

    def test_version_is_string(self):
        """__version__ is a non-empty string."""
        assert isinstance(app.__version__, str)
        assert len(app.__version__) > 0

    def test_author_is_string(self):
        """__author__ is a non-empty string."""
        assert isinstance(app.__author__, str)
        assert len(app.__author__) > 0

    def test_description_is_string(self):
        """__description__ is a non-empty string."""
        assert isinstance(app.__description__, str)
        assert len(app.__description__) > 0

    def test_version_semver_like(self):
        """__version__ has at least two dot-separated parts."""
        parts = app.__version__.split(".")
        assert len(parts) >= 2

    def test_all_exports_present(self):
        """All names in __all__ are importable from app."""
        for name in app.__all__:
            assert hasattr(app, name)
