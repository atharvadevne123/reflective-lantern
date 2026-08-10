"""Grid region registry for Watt-Guard."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

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
@lru_cache(maxsize=64)
def get_region(region_id: str) -> RegionDict | None:
    """Return the region metadata dict for *region_id*, or None if unknown."""
    return KNOWN_REGIONS.get(region_id.lower())


def list_regions() -> list[RegionDict]:
    """Return all known regions with their id injected."""
    return [{"id": k, **v} for k, v in KNOWN_REGIONS.items()]


@lru_cache(maxsize=64)
def validate_region(region_id: str) -> bool:
    """Return True if *region_id* is a recognised grid region."""
    return region_id.lower() in KNOWN_REGIONS


@lru_cache(maxsize=1)
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
        logger.debug("get_peak_load: unknown region_id=%r", region_id)
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
    """Return the total number of known regions.

    Returns:
        Integer count of regions in the registry.
    """
    return len(list_regions())


def get_region_names() -> list[str]:
    """Return a list of human-readable region names.

    Returns:
        Sorted list of region name strings.
    """
    return sorted(r.get("name", r["id"]) for r in list_regions())


def regions_by_peak_load(descending: bool = True) -> list[str]:
    """Return region IDs sorted by peak load capacity.

    Args:
        descending: If True, highest peak load first. Default True.

    Returns:
        List of region ID strings sorted by peak load.
    """
    regions = [r for r in list_regions() if r.get("peak_load_mw") is not None]
    regions.sort(key=lambda r: r["peak_load_mw"], reverse=descending)  # type: ignore[index]
    return [r["id"] for r in regions]


def region_ids_for_timezones(timezones: list[str]) -> list[str]:
    """Return region IDs that match any of the given timezones.

    Args:
        timezones: List of timezone strings to match.

    Returns:
        Sorted list of matching region IDs.
    """
    tz_set = set(timezones)
    return sorted(r["id"] for r in list_regions() if r.get("timezone") in tz_set)


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


def total_peak_load_mw() -> float:
    """Return the sum of peak loads (in MW) across all known regions.

    Returns:
        Total peak load in MW.
    """
    return sum(
        float(r["peak_load_mw"])
        for r in KNOWN_REGIONS.values()
        if r.get("peak_load_mw") is not None
    )


def regions_above_peak(threshold_mw: float) -> list[str]:
    """Return IDs of regions whose peak load exceeds *threshold_mw*.

    Args:
        threshold_mw: Minimum peak load in MW (exclusive).

    Returns:
        Sorted list of region IDs with peak load > threshold.
    """
    return sorted(
        region_id
        for region_id, r in KNOWN_REGIONS.items()
        if r.get("peak_load_mw") is not None and float(r["peak_load_mw"]) > threshold_mw
    )


def region_share_of_total(region_id: str) -> float:
    """Return this region's peak load as a fraction of total peak load.

    Args:
        region_id: Region identifier.

    Returns:
        Fraction in [0, 1]; 0.0 if region is unknown or total is zero.
    """
    region = get_region(region_id)
    if region is None or region.get("peak_load_mw") is None:
        return 0.0
    total = total_peak_load_mw()
    if total == 0.0:
        return 0.0
    return round(float(region["peak_load_mw"]) / total, 6)
