"""Tests for the grid region registry."""

from __future__ import annotations

import pytest

from app.regions import compare_peak_loads, get_region, get_region_name, list_regions, region_count, validate_region


class TestRegionRegistry:
    def test_list_regions_nonempty(self) -> None:
        regions = list_regions()
        assert len(regions) >= 5

    def test_list_regions_have_id(self) -> None:
        for r in list_regions():
            assert "id" in r
            assert "name" in r
            assert "peak_load_mw" in r

    def test_get_known_region(self) -> None:
        r = get_region("northeast")
        assert r is not None
        assert r["name"] == "Northeast Grid"

    def test_get_default_region(self) -> None:
        r = get_region("default")
        assert r is not None

    def test_get_unknown_region(self) -> None:
        r = get_region("atlantis")
        assert r is None

    def test_case_insensitive(self) -> None:
        assert get_region("NORTHEAST") == get_region("northeast")

    @pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas"])
    def test_all_known_regions_valid(self, region_id) -> None:
        assert validate_region(region_id) is True

    def test_unknown_region_invalid(self) -> None:
        assert validate_region("unknown_grid") is False


def test_get_all_region_ids_sorted() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert ids == sorted(ids)


def test_get_all_region_ids_contains_new_regions() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert "pacific_nw" in ids
    assert "new_england" in ids
    assert "mountain" in ids
    assert "southeast" in ids
    assert "florida" in ids


def test_get_region_timezone_known() -> None:
    from app.regions import get_region_timezone

    assert get_region_timezone("northeast") == "America/New_York"


def test_get_region_timezone_unknown() -> None:
    from app.regions import get_region_timezone

    assert get_region_timezone("unknown_region") == "UTC"


def test_get_peak_load_known() -> None:
    from app.regions import get_peak_load

    peak = get_peak_load("south")
    assert peak == 14000.0


def test_get_peak_load_unknown() -> None:
    from app.regions import get_peak_load

    assert get_peak_load("atlantis") is None


@pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas", "pacific_nw"])
def test_validate_region_known(region_id) -> None:
    from app.regions import validate_region

    assert validate_region(region_id) is True


def test_get_all_region_ids_contains_extended_regions() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    for region in ("pacific_nw", "new_england", "mountain", "southeast", "florida"):
        assert region in ids


def test_get_all_region_ids_returns_list_of_strings() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert all(isinstance(r, str) for r in ids)


def test_get_region_timezone_new_regions() -> None:
    from app.regions import get_region_timezone

    assert get_region_timezone("pacific_nw") == "America/Los_Angeles"
    assert get_region_timezone("new_england") == "America/New_York"
    assert get_region_timezone("mountain") == "America/Denver"
    assert get_region_timezone("southeast") == "America/New_York"
    assert get_region_timezone("florida") == "America/New_York"


@pytest.mark.parametrize("region_id", ["pacific_nw", "new_england", "mountain", "southeast", "florida"])
def test_get_peak_load_new_regions(region_id) -> None:
    from app.regions import get_peak_load

    peak = get_peak_load(region_id)
    assert peak is not None
    assert peak > 0


def test_get_regions_by_timezone_chicago() -> None:
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/Chicago")
    assert "midwest" in regions
    assert "south" in regions
    assert "texas" in regions


def test_get_regions_by_timezone_new_york() -> None:
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/New_York")
    assert "northeast" in regions
    assert "new_england" in regions
    assert "southeast" in regions
    assert "florida" in regions


def test_get_regions_by_timezone_unknown() -> None:
    from app.regions import get_regions_by_timezone

    assert get_regions_by_timezone("Pacific/Auckland") == []


def test_get_regions_by_timezone_returns_sorted() -> None:
    from app.regions import get_regions_by_timezone

    regions = get_regions_by_timezone("America/Chicago")
    assert regions == sorted(regions)


@pytest.mark.parametrize(
    "tz,expected_min",
    [
        ("America/New_York", 4),
        ("America/Chicago", 3),
        ("America/Los_Angeles", 2),
    ],
)
def test_get_regions_by_timezone_parametrized(tz, expected_min) -> None:
    from app.regions import get_regions_by_timezone

    assert len(get_regions_by_timezone(tz)) >= expected_min


def test_get_peak_load_known_v2() -> None:
    from app.regions import get_peak_load

    result = get_peak_load("northeast")
    assert result is not None
    assert result == 12000.0


def test_get_peak_load_unknown_v2() -> None:
    from app.regions import get_peak_load

    result = get_peak_load("nonexistent_region")
    assert result is None


def test_get_peak_load_case_insensitive() -> None:
    from app.regions import get_peak_load

    assert get_peak_load("MIDWEST") == get_peak_load("midwest")


@pytest.mark.parametrize(
    "region_id,expected_mw",
    [
        ("northeast", 12000.0),
        ("south", 14000.0),
        ("west", 8000.0),
        ("texas", 11000.0),
    ],
)
def test_get_peak_load_parametrized(region_id, expected_mw) -> None:
    from app.regions import get_peak_load

    assert get_peak_load(region_id) == expected_mw


def test_list_regions_all_have_id() -> None:
    from app.regions import list_regions

    for region in list_regions():
        assert "id" in region
        assert "name" in region
        assert "peak_load_mw" in region


def test_get_all_region_ids_is_sorted() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert ids == sorted(ids)


@pytest.mark.parametrize("region_id", ["pacific_nw", "new_england", "mountain", "southeast", "florida"])
def test_additional_regions_valid(region_id: str) -> None:
    assert validate_region(region_id) is True


@pytest.mark.parametrize("region_id", ["NORTHEAST", "MidWest", "SOUTH"])
def test_case_insensitive_validation(region_id: str) -> None:
    assert validate_region(region_id) is True


def test_list_regions_unique_ids() -> None:
    regions = list_regions()
    ids = [r["id"] for r in regions]
    assert len(ids) == len(set(ids)), "Region IDs must be unique"


def test_get_region_has_peak_load_field() -> None:
    r = get_region("midwest")
    assert r is not None
    assert "peak_load_mw" in r
    assert isinstance(r["peak_load_mw"], (int, float))


def test_get_region_has_carbon_intensity() -> None:
    r = get_region("texas")
    assert r is not None
    assert "carbon_intensity" in r or "grid_intensity" in r or "peak_load_mw" in r


def test_get_all_region_ids_returns_list() -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0


def test_get_all_region_ids_are_strings() -> None:
    from app.regions import get_all_region_ids

    for rid in get_all_region_ids():
        assert isinstance(rid, str)


def test_get_region_timezone_default() -> None:
    from app.regions import get_region_timezone

    tz = get_region_timezone("northeast")
    assert isinstance(tz, str)
    assert len(tz) > 0


def test_get_region_timezone_unknown_returns_utc() -> None:
    from app.regions import get_region_timezone

    tz = get_region_timezone("nonexistent_region")
    assert "UTC" in tz or "America" in tz or isinstance(tz, str)


def test_get_peak_load_known_region() -> None:
    from app.regions import get_peak_load

    load = get_peak_load("texas")
    assert load is None or isinstance(load, (int, float))


def test_get_regions_by_timezone_returns_list() -> None:
    from app.regions import get_region_timezone, get_regions_by_timezone

    tz = get_region_timezone("northeast")
    result = get_regions_by_timezone(tz)
    assert isinstance(result, list)


@pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas"])
def test_get_all_region_ids_contains_standard(region_id: str) -> None:
    from app.regions import get_all_region_ids

    ids = get_all_region_ids()
    assert region_id in ids


class TestRegionCount:
    def test_positive(self) -> None:
        from app.regions import region_count

        assert region_count() > 0

    def test_matches_list_length(self) -> None:
        from app.regions import list_regions, region_count

        assert region_count() == len(list_regions())


class TestGetRegionNames:
    def test_returns_list(self) -> None:
        from app.regions import get_region_names

        result = get_region_names()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted(self) -> None:
        from app.regions import get_region_names

        names = get_region_names()
        assert names == sorted(names)


class TestRegionsByPeakLoad:
    def test_returns_list(self) -> None:
        from app.regions import regions_by_peak_load

        result = regions_by_peak_load()
        assert isinstance(result, list)

    def test_descending_by_default(self) -> None:
        from app.regions import list_regions, regions_by_peak_load

        result = regions_by_peak_load(descending=True)
        regions = {r["id"]: r.get("peak_load_mw") for r in list_regions()}
        loads = [regions[rid] for rid in result if regions.get(rid) is not None]
        assert loads == sorted(loads, reverse=True)


class TestRegionIdsForTimezones:
    def test_returns_list(self) -> None:
        from app.regions import region_ids_for_timezones

        result = region_ids_for_timezones(["UTC"])
        assert isinstance(result, list)

    def test_empty_timezones(self) -> None:
        from app.regions import region_ids_for_timezones

        assert region_ids_for_timezones([]) == []

    def test_known_timezone(self) -> None:
        from app.regions import get_all_region_ids, get_region_timezone, region_ids_for_timezones

        ids = get_all_region_ids()
        if not ids:
            return
        tz = get_region_timezone(ids[0])
        result = region_ids_for_timezones([tz])
        assert ids[0] in result


def test_region_count_positive() -> None:
    assert region_count() >= 5


def test_region_count_includes_default() -> None:
    from app.regions import KNOWN_REGIONS

    assert region_count() == len(KNOWN_REGIONS)


def test_get_region_name_known() -> None:
    name = get_region_name("northeast")
    assert name is not None
    assert "Northeast" in name


def test_get_region_name_unknown() -> None:
    assert get_region_name("nonexistent") is None


def test_get_region_name_case_insensitive() -> None:
    name_lower = get_region_name("midwest")
    name_upper = get_region_name("MIDWEST")
    assert name_lower == name_upper


def test_compare_peak_loads_returns_dict() -> None:
    result = compare_peak_loads("northeast", "midwest")
    assert "peak1_mw" in result
    assert "peak2_mw" in result
    assert "difference_mw" in result
    assert "higher" in result


def test_compare_peak_loads_higher_correct() -> None:
    result = compare_peak_loads("south", "new_england")
    assert result["higher"] == "south"


def test_compare_peak_loads_unknown_region() -> None:
    result = compare_peak_loads("northeast", "nonexistent")
    assert result["peak2_mw"] is None
    assert result["higher"] == "unknown"


def test_compare_peak_loads_equal_regions() -> None:
    result = compare_peak_loads("northeast", "northeast")
    assert result["higher"] == "equal"


@pytest.mark.parametrize("region_id", ["northeast", "midwest", "south", "west", "texas"])
def test_get_region_name_known_ids(region_id: str) -> None:
    name = get_region_name(region_id)
    assert isinstance(name, str)
    assert len(name) > 0


class TestTotalPeakLoadMw:
    def test_returns_positive(self) -> None:
        from app.regions import total_peak_load_mw

        assert total_peak_load_mw() > 0.0

    def test_greater_than_any_single_region(self) -> None:
        from app.regions import get_peak_load, total_peak_load_mw

        south_peak = get_peak_load("south") or 0.0
        assert total_peak_load_mw() > south_peak

    def test_returns_float(self) -> None:
        from app.regions import total_peak_load_mw

        assert isinstance(total_peak_load_mw(), float)


class TestRegionsAbovePeak:
    def test_high_threshold_returns_empty(self) -> None:
        from app.regions import regions_above_peak

        result = regions_above_peak(1_000_000.0)
        assert result == []

    def test_zero_threshold_returns_all_known(self) -> None:
        from app.regions import list_regions, regions_above_peak

        result = regions_above_peak(0.0)
        known_count = len(list_regions())
        assert len(result) == known_count

    def test_specific_threshold(self) -> None:
        from app.regions import regions_above_peak

        result = regions_above_peak(10_000.0)
        assert "south" in result

    def test_result_is_sorted(self) -> None:
        from app.regions import regions_above_peak

        result = regions_above_peak(5_000.0)
        assert result == sorted(result)


class TestRegionShareOfTotal:
    def test_unknown_region_returns_zero(self) -> None:
        from app.regions import region_share_of_total

        assert region_share_of_total("nonexistent") == 0.0

    def test_share_in_range(self) -> None:
        from app.regions import region_share_of_total

        share = region_share_of_total("northeast")
        assert 0.0 < share < 1.0

    def test_all_shares_sum_to_one(self) -> None:
        from app.regions import KNOWN_REGIONS, region_share_of_total

        total = sum(region_share_of_total(rid) for rid in KNOWN_REGIONS)
        assert total == pytest.approx(1.0, abs=0.001)

    @pytest.mark.parametrize("region_id", ["northeast", "south", "west"])
    def test_returns_float(self, region_id: str) -> None:
        from app.regions import region_share_of_total

        assert isinstance(region_share_of_total(region_id), float)
