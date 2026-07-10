"""Grid region registry for Volt-Cast."""

from __future__ import annotations

KNOWN_REGIONS: dict[str, dict] = {
    "northeast": {"name": "Northeast Grid", "peak_load_mw": 12000, "timezone": "America/New_York"},
    "midwest": {"name": "Midwest Grid", "peak_load_mw": 9500, "timezone": "America/Chicago"},
    "south": {"name": "Southern Grid", "peak_load_mw": 14000, "timezone": "America/Chicago"},
    "west": {"name": "Western Grid", "peak_load_mw": 8000, "timezone": "America/Los_Angeles"},
    "texas": {"name": "ERCOT (Texas)", "peak_load_mw": 11000, "timezone": "America/Chicago"},
    "default": {"name": "Default Region", "peak_load_mw": 10000, "timezone": "UTC"},
}


def get_region(region_id: str) -> dict | None:
    return KNOWN_REGIONS.get(region_id.lower())


def list_regions() -> list[dict]:
    return [{"id": k, **v} for k, v in KNOWN_REGIONS.items()]


def validate_region(region_id: str) -> bool:
    return region_id.lower() in KNOWN_REGIONS
