"""Grid region registry for Watt-Guard."""

from __future__ import annotations

from functools import lru_cache

RegionDict = dict[str, object]

KNOWN_REGIONS: dict[str, RegionDict] = {
    "northeast": {"name": "Northeast Grid", "peak_load_mw": 12000, "timezone": "America/New_York"},
    "midwest": {"name": "Midwest Grid", "peak_load_mw": 9500, "timezone": "America/Chicago"},
    "south": {"name": "Southern Grid", "peak_load_mw": 14000, "timezone": "America/Chicago"},
    "west": {"name": "Western Grid", "peak_load_mw": 8000, "timezone": "America/Los_Angeles"},
    "texas": {"name": "ERCOT (Texas)", "peak_load_mw": 11000, "timezone": "America/Chicago"},
    "pacific_nw": {"name": "Pacific Northwest Grid", "peak_load_mw": 7200, "timezone": "America/Los_Angeles"},
    "new_england": {"name": "New England Grid (ISO-NE)", "peak_load_mw": 5500, "timezone": "America/New_York"},
    "mountain": {"name": "Mountain States Grid", "peak_load_mw": 6800, "timezone": "America/Denver"},
    "southeast": {"name": "Southeast Grid (SERC)", "peak_load_mw": 10500, "timezone": "America/New_York"},
    "florida": {"name": "Florida Grid (FRCC)", "peak_load_mw": 7000, "timezone": "America/New_York"},
    "default": {"name": "Default Region", "peak_load_mw": 10000, "timezone": "UTC"},
}


@lru_cache(maxsize=64)
def get_region(region_id: str) -> RegionDict | None:
    """Return the region metadata dict for *region_id*, or None if unknown."""
    return KNOWN_REGIONS.get(region_id.lower())


def list_regions() -> list[RegionDict]:
    """Return all known regions with their id injected."""
    return [{"id": k, **v} for k, v in KNOWN_REGIONS.items()]


def validate_region(region_id: str) -> bool:
    """Return True if *region_id* is a recognised grid region."""
    return region_id.lower() in KNOWN_REGIONS


def get_all_region_ids() -> list[str]:
    """Return sorted list of all registered grid region identifiers."""
    return sorted(KNOWN_REGIONS.keys())


def get_region_timezone(region_id: str) -> str:
    """Return the IANA timezone string for *region_id*, defaulting to UTC.

    Args:
        region_id: Grid region identifier (case-insensitive).

    Returns:
        IANA timezone string (e.g. 'America/Chicago').
    """
    region = get_region(region_id)
    if region is None:
        return "UTC"
    return str(region.get("timezone", "UTC"))


def get_regions_by_timezone(timezone: str) -> list[str]:
    """Return all region IDs that share the given IANA *timezone*.

    Args:
        timezone: IANA timezone string to filter by (e.g. 'America/Chicago').

    Returns:
        Sorted list of region IDs whose timezone matches (case-sensitive).
    """
    return sorted(region_id for region_id, meta in KNOWN_REGIONS.items() if str(meta.get("timezone", "")) == timezone)


def get_peak_load(region_id: str) -> float | None:
    """Return the nominal peak load in MW for *region_id*, or None if unknown.

    Args:
        region_id: Grid region identifier (case-insensitive).

    Returns:
        Peak load in megawatts, or None for unrecognised regions.
    """
    region = get_region(region_id)
    if region is None:
        return None
    peak = region.get("peak_load_mw")
    return float(peak) if peak is not None else None

__all__ = [
    "compare_peak_loads",
    "get_all_region_ids",
    "get_peak_load",
    "get_region",
    "get_region_name",
    "get_region_timezone",
    "get_regions_by_timezone",
    "list_regions",
    "region_count",
    "validate_region",
]


def region_count() -> int:
    """Return the total number of registered grid regions.

    Returns:
        Integer count of entries in KNOWN_REGIONS.
    """
    return len(KNOWN_REGIONS)


def get_region_name(region_id: str) -> str | None:
    """Return the human-readable name for *region_id*, or None if unknown.

    Args:
        region_id: Grid region identifier (case-insensitive).

    Returns:
        Name string such as 'Northeast Grid', or None for unrecognised IDs.
    """
    region = get_region(region_id)
    if region is None:
        return None
    name = region.get("name")
    return str(name) if name is not None else None


def compare_peak_loads(region_id1: str, region_id2: str) -> dict[str, object]:
    """Compare peak load values for two grid regions.

    Args:
        region_id1: First region identifier.
        region_id2: Second region identifier.

    Returns:
        Dict with 'region1', 'region2', 'peak1_mw', 'peak2_mw',
        'difference_mw', and 'higher' (the ID of the higher-load region,
        or 'equal' when identical). Unknown regions have None peak values.
    """
    peak1 = get_peak_load(region_id1)
    peak2 = get_peak_load(region_id2)
    diff: float | None = None
    if peak1 is not None and peak2 is not None:
        diff = round(peak1 - peak2, 2)
    if peak1 is not None and peak2 is not None:
        if peak1 > peak2:
            higher: str = region_id1
        elif peak2 > peak1:
            higher = region_id2
        else:
            higher = "equal"
    else:
        higher = "unknown"
    return {
        "region1": region_id1,
        "region2": region_id2,
        "peak1_mw": peak1,
        "peak2_mw": peak2,
        "difference_mw": diff,
        "higher": higher,
    }
