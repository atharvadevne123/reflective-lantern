"""Input validation helpers shared by the API and batch ingestion paths."""

from __future__ import annotations

from typing import Any

FAULT_TYPES = ("strike_slip", "reverse", "normal", "oblique", "unknown")

REQUIRED_FIELDS = (
    "latitude",
    "longitude",
    "depth_km",
    "station_count",
    "p_wave_amplitude",
    "s_wave_amplitude",
    "epicentral_distance_km",
    "fault_type",
)

RANGES: dict[str, tuple[float, float]] = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "depth_km": (0.0, 700.0),
    "station_count": (1, 500),
    "p_wave_amplitude": (0.0, 1e6),
    "s_wave_amplitude": (0.0, 1e6),
    "epicentral_distance_km": (0.0, 20000.0),
}


def missing_fields(payload: dict[str, Any]) -> list[str]:
    """Return the required fields absent from ``payload``."""
    return [field for field in REQUIRED_FIELDS if field not in payload]


def out_of_range_fields(payload: dict[str, Any]) -> list[str]:
    """Return numeric fields whose values fall outside their physical range."""
    bad = []
    for field, (low, high) in RANGES.items():
        value = payload.get(field)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not (low <= float(value) <= high):
            bad.append(field)
    return bad


def normalise_fault_type(value: str) -> str:
    """Lowercase and trim a fault type, falling back to ``unknown``."""
    cleaned = str(value).lower().strip().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in FAULT_TYPES else "unknown"


def amplitudes_are_coherent(p_wave: float, s_wave: float) -> bool:
    """Whether an S/P amplitude pair is physically plausible.

    S-waves carry more energy than P-waves from the same source, so an S
    amplitude below half the P amplitude almost always indicates transposed or
    mis-parsed columns rather than a genuine recording.
    """
    if p_wave <= 0 or s_wave <= 0:
        return False
    return s_wave >= p_wave * 0.5


def validate_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an event payload, returning a structured verdict.

    Returns:
        A dict with ``valid`` plus lists of ``missing``, ``out_of_range`` and
        free-form ``warnings``.
    """
    missing = missing_fields(payload)
    out_of_range = out_of_range_fields(payload)
    warnings: list[str] = []

    p_wave = payload.get("p_wave_amplitude")
    s_wave = payload.get("s_wave_amplitude")
    if (
        isinstance(p_wave, (int, float))
        and isinstance(s_wave, (int, float))
        and not amplitudes_are_coherent(float(p_wave), float(s_wave))
    ):
        warnings.append("S/P amplitude ratio is implausible — check for transposed columns")

    fault_type = payload.get("fault_type")
    if (
        fault_type is not None
        and normalise_fault_type(str(fault_type)) == "unknown"
        and str(fault_type).lower().strip() != "unknown"
    ):
        warnings.append(f"Unrecognised fault_type {fault_type!r} — treated as 'unknown'")

    return {
        "valid": not missing and not out_of_range,
        "missing": missing,
        "out_of_range": out_of_range,
        "warnings": warnings,
    }
