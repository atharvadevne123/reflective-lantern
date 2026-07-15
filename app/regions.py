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
