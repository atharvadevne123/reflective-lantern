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


def test_get_all_region_ids_sorted():
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert ids == sorted(ids)


def test_get_all_region_ids_contains_new_regions():
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert "pacific_nw" in ids
    assert "new_england" in ids
    assert "mountain" in ids
    assert "southeast" in ids
    assert "florida" in ids


def test_get_region_timezone_known():
    from app.regions import get_region_timezone

    assert get_region_timezone("northeast") == "America/New_York"


def test_get_region_timezone_unknown():
    from app.regions import get_region_timezone

    assert get_region_timezone("unknown_region") == "UTC"


def test_get_peak_load_known():
    from app.regions import get_peak_load

    peak = get_peak_load("south")
    assert peak == 14000.0


def test_get_peak_load_unknown():
    from app.regions import get_peak_load

    assert get_peak_load("atlantis") is None


@pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas", "pacific_nw"])
def test_validate_region_known(region_id):
    from app.regions import validate_region

    assert validate_region(region_id) is True


def test_get_all_region_ids_contains_extended_regions():
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    for region in ("pacific_nw", "new_england", "mountain", "southeast", "florida"):
        assert region in ids


def test_get_all_region_ids_returns_list_of_strings():
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert all(isinstance(r, str) for r in ids)


def test_get_region_timezone_new_regions():
    from app.regions import get_region_timezone

    assert get_region_timezone("pacific_nw") == "America/Los_Angeles"
    assert get_region_timezone("new_england") == "America/New_York"
    assert get_region_timezone("mountain") == "America/Denver"
    assert get_region_timezone("southeast") == "America/New_York"
    assert get_region_timezone("florida") == "America/New_York"


@pytest.mark.parametrize("region_id", ["pacific_nw", "new_england", "mountain", "southeast", "florida"])
def test_get_peak_load_new_regions(region_id):
    from app.regions import get_peak_load

    peak = get_peak_load(region_id)
    assert peak is not None
    assert peak > 0


def test_get_regions_by_timezone_chicago():
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/Chicago")
    assert "midwest" in regions
    assert "south" in regions
    assert "texas" in regions


def test_get_regions_by_timezone_new_york():
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/New_York")
    assert "northeast" in regions
    assert "new_england" in regions
    assert "southeast" in regions
    assert "florida" in regions


def test_get_regions_by_timezone_unknown():
    from app.regions import get_regions_by_timezone

    assert get_regions_by_timezone("Pacific/Auckland") == []


def test_get_regions_by_timezone_returns_sorted():
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/Chicago")
    assert regions == sorted(regions)


@pytest.mark.parametrize("tz,expected_min", [
    ("America/New_York", 4),
    ("America/Chicago", 3),
    ("America/Los_Angeles", 2),
])
def test_get_regions_by_timezone_parametrized(tz, expected_min):
    from app.regions import get_regions_by_timezone

    assert len(get_regions_by_timezone(tz)) >= expected_min


def test_get_peak_load_known_v2():
    from app.regions import get_peak_load

    result = get_peak_load("northeast")
    assert result is not None
    assert result == 12000.0


def test_get_peak_load_unknown_v2():
    from app.regions import get_peak_load

    result = get_peak_load("nonexistent_region")
    assert result is None


def test_get_peak_load_case_insensitive():
    from app.regions import get_peak_load

    assert get_peak_load("MIDWEST") == get_peak_load("midwest")


@pytest.mark.parametrize("region_id,expected_mw", [
    ("northeast", 12000.0),
    ("south", 14000.0),
    ("west", 8000.0),
    ("texas", 11000.0),
])
def test_get_peak_load_parametrized(region_id, expected_mw):
    from app.regions import get_peak_load

    assert get_peak_load(region_id) == expected_mw


def test_list_regions_all_have_id():
    from app.regions import list_regions

    for region in list_regions():
        assert "id" in region
        assert "name" in region
        assert "peak_load_mw" in region


def test_get_all_region_ids_is_sorted():
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert ids == sorted(ids)
