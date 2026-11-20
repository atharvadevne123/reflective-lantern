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
    "region_load_factor",
    "region_names",
    "region_share_of_total",
    "regions_above_peak",
    "regions_by_grid_type",
    "regions_by_peak_load",
    "regions_summary",
    "total_peak_load_mw",
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


def regions_summary() -> dict[str, object]:
    """Return an aggregate summary of all registered regions.

    Returns:
        Dict with 'total_regions', 'timezones' (unique set), 'total_peak_load_mw',
        and 'max_peak_region' (region_id with highest peak load).
    """
    all_regions = list_regions()
    timezones = sorted({str(r.get("timezone", "UTC")) for r in all_regions})
    peaks = {r["id"]: float(r.get("peak_load_mw", 0)) for r in all_regions}
    total_peak = round(sum(peaks.values()), 2)
    max_region = max(peaks, key=lambda k: peaks[k]) if peaks else None
    return {
        "total_regions": len(all_regions),
        "timezones": timezones,
        "total_peak_load_mw": total_peak,
        "max_peak_region": max_region,
    }


def total_peak_load_mw() -> float:
    """Return the sum of peak loads across all known regions except 'default'.

    Returns:
        Total peak load in MW.
    """
    return float(sum(r.get("peak_load_mw", 0.0) for name, r in KNOWN_REGIONS.items() if name != "default"))


def regions_above_peak(threshold_mw: float) -> list[str]:
    """Return the ids of regions whose peak load exceeds *threshold_mw*.

    Args:
        threshold_mw: Threshold in megawatts.

    Returns:
        Sorted list of region ids (includes every entry in :data:`KNOWN_REGIONS`).
    """
    return sorted(name for name, r in KNOWN_REGIONS.items() if r.get("peak_load_mw", 0.0) > threshold_mw)


def region_share_of_total(region_id: str) -> float:
    """Return the fraction of total peak load contributed by *region_id*.

    Args:
        region_id: Region key.

    Returns:
        Fraction in [0, 1]; 0.0 when the region is unknown.
    """
    region = KNOWN_REGIONS.get(region_id.lower())
    if region is None:
        return 0.0
    total = float(sum(r.get("peak_load_mw", 0.0) for r in KNOWN_REGIONS.values()))
    if total <= 0:
        return 0.0
    return round(float(region.get("peak_load_mw", 0.0)) / total, 6)


def regions_by_grid_type(grid_type: str) -> list[str]:
    """Return region ids that match a given grid type.

    The registry does not currently store a grid_type field, so this helper
    is provided for API completeness and always returns an empty list unless
    the supplied *grid_type* matches an internal alias.

    Args:
        grid_type: Grid classification identifier.

    Returns:
        List of matching region ids (possibly empty).
    """
    aliases: dict[str, list[str]] = {
        "east": ["northeast", "new_england", "southeast", "florida"],
        "west": ["west", "pacific_nw", "mountain"],
        "central": ["midwest", "south", "texas"],
    }
    return list(aliases.get(grid_type.lower(), []))


def region_names() -> list[str]:
    """Return a sorted list of known region ids (excluding ``default``).

    Returns:
        Sorted list of region identifier strings.
    """
    return sorted(name for name in KNOWN_REGIONS if name != "default")


def region_load_factor(region_id: str, load_mw: float) -> float:
    """Return the fraction of *region_id*'s peak load used by *load_mw*.

    Args:
        region_id: Region key.
        load_mw: Current load in megawatts (must be non-negative).

    Returns:
        Fraction clamped to [0, 1]; 0.0 when the region is unknown.

    Raises:
        ValueError: If ``load_mw`` is negative.
    """
    if load_mw < 0:
        raise ValueError(f"load_mw must be non-negative, got {load_mw}")
    region = KNOWN_REGIONS.get(region_id.lower())
    if region is None:
        return 0.0
    peak = float(region.get("peak_load_mw", 0.0))
    if peak <= 0:
        return 0.0
    return round(min(1.0, load_mw / peak), 6)


def highest_peak_region() -> str | None:
    """Return the region ID with the highest peak load, or None if no regions are defined.

    Returns:
        Region ID string, or None if KNOWN_REGIONS is empty.
    """
    if not KNOWN_REGIONS:
        return None
    return max(
        KNOWN_REGIONS,
        key=lambda rid: float(KNOWN_REGIONS[rid].get("peak_load_mw", 0.0)),
    )


def lowest_peak_region() -> str | None:
    """Return the region ID with the lowest peak load, or None if no regions are defined.

    Returns:
        Region ID string, or None if KNOWN_REGIONS is empty.
    """
    if not KNOWN_REGIONS:
        return None
    return min(
        KNOWN_REGIONS,
        key=lambda rid: float(KNOWN_REGIONS[rid].get("peak_load_mw", 0.0)),
    )


def peak_load_range_mw() -> dict[str, float]:
    """Return a dict with 'min' and 'max' peak loads across all known regions.

    Returns:
        Dict with keys 'min' and 'max' in MW; both 0.0 when no regions are defined.
    """
    if not KNOWN_REGIONS:
        return {"min": 0.0, "max": 0.0}
    loads = [float(r.get("peak_load_mw", 0.0)) for r in KNOWN_REGIONS.values()]
    return {"min": min(loads), "max": max(loads)}
