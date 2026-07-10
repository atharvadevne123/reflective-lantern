"""Tests for the grid region registry."""

from __future__ import annotations

import pytest

from app.regions import get_region, list_regions, validate_region


class TestRegionRegistry:
    def test_list_regions_nonempty(self):
        regions = list_regions()
        assert len(regions) >= 5

    def test_list_regions_have_id(self):
        for r in list_regions():
            assert "id" in r
            assert "name" in r
            assert "peak_load_mw" in r

    def test_get_known_region(self):
        r = get_region("northeast")
        assert r is not None
        assert r["name"] == "Northeast Grid"

    def test_get_default_region(self):
        r = get_region("default")
        assert r is not None

    def test_get_unknown_region(self):
        r = get_region("atlantis")
        assert r is None

    def test_case_insensitive(self):
        assert get_region("NORTHEAST") == get_region("northeast")

    @pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas"])
    def test_all_known_regions_valid(self, region_id):
        assert validate_region(region_id) is True

    def test_unknown_region_invalid(self):
        assert validate_region("unknown_grid") is False
