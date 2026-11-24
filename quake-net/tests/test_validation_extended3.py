"""Extended seismic event validation tests — third pass with parametrize."""
from __future__ import annotations

import pytest

from app.validation import (
    FAULT_TYPES,
    REQUIRED_FIELDS,
    amplitudes_are_coherent,
    missing_fields,
    normalise_fault_type,
    out_of_range_fields,
    validate_event,
)


def _valid_payload() -> dict:
    return {
        "latitude": 35.0,
        "longitude": -120.0,
        "depth_km": 10.0,
        "station_count": 5,
        "p_wave_amplitude": 100.0,
        "s_wave_amplitude": 200.0,
        "epicentral_distance_km": 50.0,
        "fault_type": "strike_slip",
    }


def test_valid_payload_no_missing_fields() -> None:
    assert missing_fields(_valid_payload()) == []


def test_valid_payload_no_out_of_range() -> None:
    assert out_of_range_fields(_valid_payload()) == []


def test_valid_payload_validate_event_is_valid() -> None:
    result = validate_event(_valid_payload())
    assert result["valid"] is True


@pytest.mark.parametrize("field", list(REQUIRED_FIELDS))
def test_missing_each_required_field(field: str) -> None:
    payload = _valid_payload()
    del payload[field]
    assert field in missing_fields(payload)


def test_out_of_range_latitude_too_large() -> None:
    payload = _valid_payload()
    payload["latitude"] = 95.0
    assert "latitude" in out_of_range_fields(payload)


def test_out_of_range_depth_negative() -> None:
    payload = _valid_payload()
    payload["depth_km"] = -1.0
    assert "depth_km" in out_of_range_fields(payload)


@pytest.mark.parametrize("field,bad_value", [
    ("latitude", 95.0),
    ("longitude", 200.0),
    ("depth_km", -5.0),
    ("station_count", 600),
    ("epicentral_distance_km", 25000.0),
])
def test_out_of_range_fields_various(field: str, bad_value: float) -> None:
    payload = _valid_payload()
    payload[field] = bad_value
    assert field in out_of_range_fields(payload)


def test_amplitudes_coherent_s_larger_than_p() -> None:
    assert amplitudes_are_coherent(100.0, 200.0) is True


def test_amplitudes_not_coherent_s_too_small() -> None:
    assert amplitudes_are_coherent(100.0, 10.0) is False


def test_amplitudes_coherent_exactly_half() -> None:
    assert amplitudes_are_coherent(100.0, 50.0) is True


def test_amplitudes_coherent_zero_p_wave_false() -> None:
    assert amplitudes_are_coherent(0.0, 100.0) is False


@pytest.mark.parametrize("fault_type", list(FAULT_TYPES))
def test_normalise_fault_type_known_types(fault_type: str) -> None:
    assert normalise_fault_type(fault_type) == fault_type


def test_normalise_fault_type_case_insensitive() -> None:
    assert normalise_fault_type("Strike_Slip") == "strike_slip"


def test_normalise_fault_type_unknown_returns_unknown() -> None:
    assert normalise_fault_type("alien_fault") == "unknown"


def test_normalise_fault_type_hyphen_replaced() -> None:
    assert normalise_fault_type("strike-slip") == "strike_slip"


def test_validate_event_missing_field_not_valid() -> None:
    payload = _valid_payload()
    del payload["latitude"]
    result = validate_event(payload)
    assert result["valid"] is False
    assert "latitude" in result["missing"]


def test_validate_event_out_of_range_not_valid() -> None:
    payload = _valid_payload()
    payload["latitude"] = 95.0
    result = validate_event(payload)
    assert result["valid"] is False
    assert "latitude" in result["out_of_range"]


def test_validate_event_has_required_keys() -> None:
    result = validate_event(_valid_payload())
    for key in ("valid", "missing", "out_of_range", "warnings"):
        assert key in result
