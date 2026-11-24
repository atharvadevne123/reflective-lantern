"""Extended validation tests: parametrize ranges, fault types, amplitude coherence."""
from __future__ import annotations

import pytest

from app.validation import (
    amplitudes_are_coherent,
    missing_fields,
    normalise_fault_type,
    out_of_range_fields,
    validate_event,
)


def _valid_payload(**overrides) -> dict:
    base = {
        "latitude": 35.0,
        "longitude": -120.0,
        "depth_km": 10.0,
        "station_count": 5,
        "p_wave_amplitude": 100.0,
        "s_wave_amplitude": 200.0,
        "epicentral_distance_km": 50.0,
        "fault_type": "strike_slip",
    }
    base.update(overrides)
    return base


def test_valid_payload_passes() -> None:
    result = validate_event(_valid_payload())
    assert result["valid"] is True
    assert result["missing"] == []
    assert result["out_of_range"] == []


@pytest.mark.parametrize("field", [
    "latitude", "longitude", "depth_km", "station_count",
    "p_wave_amplitude", "s_wave_amplitude", "epicentral_distance_km", "fault_type",
])
def test_missing_required_field_detected(field: str) -> None:
    payload = _valid_payload()
    del payload[field]
    result = validate_event(payload)
    assert result["valid"] is False
    assert field in result["missing"]


def test_all_fields_missing() -> None:
    result = validate_event({})
    assert result["valid"] is False
    assert len(result["missing"]) == 8


@pytest.mark.parametrize("lat,valid", [
    (-90.0, True),
    (0.0, True),
    (90.0, True),
    (-90.1, False),
    (90.1, False),
    (180.0, False),
])
def test_latitude_range(lat: float, valid: bool) -> None:
    payload = _valid_payload(latitude=lat)
    result = validate_event(payload)
    if valid:
        assert "latitude" not in result["out_of_range"]
    else:
        assert "latitude" in result["out_of_range"]


@pytest.mark.parametrize("lon,valid", [
    (-180.0, True),
    (0.0, True),
    (180.0, True),
    (-180.1, False),
    (180.1, False),
])
def test_longitude_range(lon: float, valid: bool) -> None:
    payload = _valid_payload(longitude=lon)
    result = validate_event(payload)
    if valid:
        assert "longitude" not in result["out_of_range"]
    else:
        assert "longitude" in result["out_of_range"]


@pytest.mark.parametrize("depth,valid", [
    (0.0, True),
    (350.0, True),
    (700.0, True),
    (-1.0, False),
    (701.0, False),
])
def test_depth_km_range(depth: float, valid: bool) -> None:
    payload = _valid_payload(depth_km=depth)
    result = validate_event(payload)
    if valid:
        assert "depth_km" not in result["out_of_range"]
    else:
        assert "depth_km" in result["out_of_range"]


@pytest.mark.parametrize("fault,expected", [
    ("strike_slip", "strike_slip"),
    ("REVERSE", "reverse"),
    ("Normal", "normal"),
    ("OBLIQUE", "oblique"),
    ("unknown", "unknown"),
    ("Strike Slip", "strike_slip"),
    ("strike-slip", "strike_slip"),
    ("garbage", "unknown"),
    ("", "unknown"),
])
def test_normalise_fault_type(fault: str, expected: str) -> None:
    assert normalise_fault_type(fault) == expected


@pytest.mark.parametrize("fault_type", ["strike_slip", "reverse", "normal", "oblique", "unknown"])
def test_valid_fault_types_no_warning(fault_type: str) -> None:
    result = validate_event(_valid_payload(fault_type=fault_type))
    assert not any("fault_type" in w for w in result["warnings"])


def test_unrecognised_fault_type_adds_warning() -> None:
    result = validate_event(_valid_payload(fault_type="volcanic"))
    assert any("fault_type" in w.lower() for w in result["warnings"])


@pytest.mark.parametrize("p_wave,s_wave,coherent", [
    (100.0, 200.0, True),
    (100.0, 50.0, True),
    (100.0, 49.0, False),
    (100.0, 0.0, False),
    (0.0, 100.0, False),
    (0.0, 0.0, False),
    (1.0, 0.5, True),
    (1.0, 0.49, False),
])
def test_amplitudes_are_coherent(p_wave: float, s_wave: float, coherent: bool) -> None:
    assert amplitudes_are_coherent(p_wave, s_wave) is coherent


def test_incoherent_amplitudes_adds_warning() -> None:
    result = validate_event(_valid_payload(p_wave_amplitude=200.0, s_wave_amplitude=50.0))
    assert any("amplitude" in w.lower() for w in result["warnings"])


def test_validate_event_returns_dict_keys() -> None:
    result = validate_event(_valid_payload())
    assert "valid" in result
    assert "missing" in result
    assert "out_of_range" in result
    assert "warnings" in result


def test_missing_fields_empty_on_complete_payload() -> None:
    assert missing_fields(_valid_payload()) == []


def test_out_of_range_fields_empty_on_valid_payload() -> None:
    assert out_of_range_fields(_valid_payload()) == []
